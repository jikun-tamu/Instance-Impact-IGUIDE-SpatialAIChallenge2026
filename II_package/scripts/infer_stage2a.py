#!/usr/bin/env python3
"""Standalone Stage-2a inference over per-building crops and masks.

This script mirrors the model architecture from stage2a/building_population_model.ipynb:
- EfficientNet-B0 backbone
- 4-channel input (RGB + mask)
- multi-task heads:
  - population regression on log1p scale
  - building-type classification logits
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CLASS_NAMES = ["residential_small", "residential_multi", "commercial", "institutional", "other"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    p = argparse.ArgumentParser(description="Run Stage-2a model inference.")
    p.add_argument("--input_csv", type=Path, required=True, help="CSV with building_uid,crop_path,mask_path")
    p.add_argument("--ckpt", type=Path, required=True, help="Path to Stage-2a checkpoint")
    p.add_argument("--out_csv", type=Path, required=True, help="Output CSV path")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--limit", type=int, default=0, help="Optional row limit")
    p.add_argument("--print_examples", type=int, default=10)
    p.add_argument("--log_every_steps", type=int, default=20)
    p.add_argument("--crop_col", type=str, default="crop_path")
    p.add_argument("--mask_col", type=str, default="mask_path")
    p.add_argument("--id_col", type=str, default="building_uid")
    return p.parse_args()


class Stage2aInferenceDataset(Dataset):
    def __init__(self, rows, img_size=224, crop_col="crop_path", mask_col="mask_path", id_col="building_uid"):
        self.rows = rows
        self.crop_col = crop_col
        self.mask_col = mask_col
        self.id_col = id_col
        self.rgb_tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.mask_tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),  # [0,255] -> [0,1]
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        crop_path = row[self.crop_col]
        mask_path = row[self.mask_col]
        rgb = Image.open(crop_path).convert("RGB")
        m = Image.open(mask_path).convert("L")
        x = torch.cat([self.rgb_tf(rgb), self.mask_tf(m)], dim=0)
        meta = {
            "building_uid": row.get(self.id_col, ""),
            "crop_path": crop_path,
            "mask_path": mask_path,
            "tile_id": row.get("tile_id", ""),
            "event_id": row.get("event_id", ""),
            "sam3_confidence": row.get("sam3_confidence", ""),
            "hazard_type": row.get("hazard_type", ""),
        }
        return x, meta


class BuildingPopulationModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=False):
        super().__init__()
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )

        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            4,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = 0.0
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        backbone.features[0][0] = new_conv

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = backbone.classifier[1].in_features
        self.shared_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
        )
        self.pop_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.type_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.shared_fc(x)
        pop_logpop = self.pop_head(x).squeeze(-1)
        type_logits = self.type_head(x)
        return pop_logpop, type_logits


def read_rows(path, limit=0):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def _load_state_dict_from_ckpt(path):
    ckpt = torch.load(path, map_location="cpu")
    # Handles common wrappers, while remaining compatible with pure state_dict saves.
    if isinstance(ckpt, dict):
        for k in ["model_state", "state_dict", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _collate(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    metas = [b[1] for b in batch]
    return xs, metas


def main():
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {args.input_csv}")
    if not args.ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.ckpt}")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.input_csv, limit=args.limit)
    if len(rows) == 0:
        raise RuntimeError("No rows loaded from input CSV.")
    for c in [args.id_col, args.crop_col, args.mask_col]:
        if c not in rows[0]:
            raise KeyError(f"Missing required column '{c}' in input CSV.")

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("[info] rows:", len(rows))
    print("[info] device:", device)
    print("[info] ckpt:", args.ckpt)

    ds = Stage2aInferenceDataset(
        rows,
        img_size=args.img_size,
        crop_col=args.crop_col,
        mask_col=args.mask_col,
        id_col=args.id_col,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        collate_fn=_collate,
    )

    model = BuildingPopulationModel(num_classes=len(CLASS_NAMES), pretrained=False).to(device)
    state = _load_state_dict_from_ckpt(args.ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    out_rows = []
    step = 0
    with torch.no_grad():
        for x, metas in loader:
            step += 1
            x = x.to(device, non_blocking=use_cuda)
            pop_log, type_logits = model(x)
            probs = torch.softmax(type_logits, dim=1).cpu().numpy()
            pop_log_np = pop_log.cpu().numpy()
            pop_np = np.expm1(pop_log_np)
            cls_idx = np.argmax(probs, axis=1)
            cls_conf = probs[np.arange(len(cls_idx)), cls_idx]

            for i, meta in enumerate(metas):
                row = {
                    "building_uid": meta["building_uid"],
                    "pred_population": float(pop_np[i]),
                    "pred_log1p_population": float(pop_log_np[i]),
                    "pred_type_idx": int(cls_idx[i]),
                    "pred_type_class": CLASS_NAMES[int(cls_idx[i])],
                    "pred_type_conf": float(cls_conf[i]),
                    "prob_residential_small": float(probs[i, 0]),
                    "prob_residential_multi": float(probs[i, 1]),
                    "prob_commercial": float(probs[i, 2]),
                    "prob_institutional": float(probs[i, 3]),
                    "prob_other": float(probs[i, 4]),
                    "crop_path": meta.get("crop_path", ""),
                    "mask_path": meta.get("mask_path", ""),
                    "tile_id": meta.get("tile_id", ""),
                    "event_id": meta.get("event_id", ""),
                    "sam3_confidence": meta.get("sam3_confidence", ""),
                    "hazard_type": meta.get("hazard_type", ""),
                }
                out_rows.append(row)

            if step % max(1, args.log_every_steps) == 0:
                print(f"[infer_stage2a] step={step}/{len(loader)} done_rows={len(out_rows)}")

    fields = [
        "building_uid",
        "pred_population",
        "pred_log1p_population",
        "pred_type_idx",
        "pred_type_class",
        "pred_type_conf",
        "prob_residential_small",
        "prob_residential_multi",
        "prob_commercial",
        "prob_institutional",
        "prob_other",
        "crop_path",
        "mask_path",
        "tile_id",
        "event_id",
        "sam3_confidence",
        "hazard_type",
    ]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    print("[done] wrote:", args.out_csv)
    print("[done] rows:", len(out_rows))
    # Quick summaries for terminal sanity.
    cls_counts = {}
    pops = []
    for r in out_rows:
        c = r["pred_type_class"]
        cls_counts[c] = cls_counts.get(c, 0) + 1
        pops.append(float(r["pred_population"]))
    print("[summary] class_counts:", cls_counts)
    if pops:
        q = np.percentile(np.asarray(pops, dtype=np.float64), [0, 25, 50, 75, 95, 99])
        print("[summary] pred_population quantiles p0/p25/p50/p75/p95/p99:", [round(float(x), 3) for x in q])

    n_show = min(max(0, args.print_examples), len(out_rows))
    if n_show > 0:
        print(f"[examples] first_{n_show}:")
        for r in out_rows[:n_show]:
            print(
                "  ",
                r["building_uid"],
                "pop=",
                round(float(r["pred_population"]), 3),
                "type=",
                r["pred_type_class"],
                "conf=",
                round(float(r["pred_type_conf"]), 4),
            )


if __name__ == "__main__":
    main()
