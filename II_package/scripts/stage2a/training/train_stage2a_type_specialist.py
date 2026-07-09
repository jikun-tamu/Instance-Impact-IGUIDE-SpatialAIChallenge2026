#!/usr/bin/env python3
"""Train Stage-2a parcel-code minority specialists and evaluate a cascade."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from scripts.stage2a.common import (
        CLASS_NAMES,
        CLASS_TO_IDX,
        DEFAULT_GEOMETRY_COLS,
        PROB_COLUMNS,
        SUPPORTED_STAGE2A_INPUT_MODES,
        SUPPORTED_STAGE2A_POOLING_MODES,
        Stage2aDataset,
        build_model_from_config,
        collate_stage2a,
        load_torch_checkpoint,
        parse_list,
        read_csv_rows,
        safe_float,
        safe_int,
        save_json,
        seed_all,
        tile_train_val_test_split,
        write_csv_rows,
    )
except ImportError:  # pragma: no cover
    from II_package.scripts.stage2a.common import (
        CLASS_NAMES,
        CLASS_TO_IDX,
        DEFAULT_GEOMETRY_COLS,
        PROB_COLUMNS,
        SUPPORTED_STAGE2A_INPUT_MODES,
        SUPPORTED_STAGE2A_POOLING_MODES,
        Stage2aDataset,
        build_model_from_config,
        collate_stage2a,
        load_torch_checkpoint,
        parse_list,
        read_csv_rows,
        safe_float,
        safe_int,
        save_json,
        seed_all,
        tile_train_val_test_split,
        write_csv_rows,
    )


def _preparse_config() -> Optional[Path]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=Path, default=None)
    args, _ = p.parse_known_args()
    return args.config


def parse_args() -> argparse.Namespace:
    cfg_path = _preparse_config()
    defaults: Dict[str, object] = {}
    if cfg_path:
        defaults = json.loads(cfg_path.read_text(encoding="utf-8"))

    p = argparse.ArgumentParser(description="Train binary minority specialists and evaluate a validation-only cascade.")
    p.add_argument("--config", type=Path, default=cfg_path)
    p.add_argument("--labels_csv", type=Path, required=not bool(defaults), default=defaults.get("labels_csv"))
    p.add_argument("--generalist_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--target_classes", type=str, default="institutional,other")
    p.add_argument("--cascade_sources", type=str, default="parcel_code")
    p.add_argument("--seed", type=int, default=int(defaults.get("seed", 2025)))
    p.add_argument("--split_seed", type=int, default=int(defaults.get("split_seed", defaults.get("seed", 2025))))
    p.add_argument("--split_group_col", type=str, default=str(defaults.get("split_group_col", "GEOID")))
    p.add_argument("--val_ratio", type=float, default=float(defaults.get("val_ratio", 0.15)))
    p.add_argument("--test_ratio", type=float, default=float(defaults.get("test_ratio", 0.15)))
    p.add_argument("--epochs", type=int, default=int(defaults.get("epochs", 40)))
    p.add_argument("--batch_size", type=int, default=int(defaults.get("batch_size", 48)))
    p.add_argument("--lr", type=float, default=float(defaults.get("lr", 5e-5)))
    p.add_argument("--weight_decay", type=float, default=float(defaults.get("weight_decay", 0.01)))
    p.add_argument("--early_stop_patience", type=int, default=int(defaults.get("early_stop_patience", 8)))
    p.add_argument("--amp_dtype", choices=["bf16", "fp16", "off"], default=str(defaults.get("amp_dtype", "bf16")))
    p.add_argument("--binary_loss_mode", choices=["bce", "focal_bce"], default="focal_bce")
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--pos_weight_cap", type=float, default=20.0)
    p.add_argument("--threshold_min", type=float, default=0.20)
    p.add_argument("--threshold_max", type=float, default=0.90)
    p.add_argument("--threshold_step", type=float, default=0.05)
    p.add_argument("--max_val_accuracy_drop", type=float, default=0.03)
    p.add_argument("--max_area_absent_fp_increase", type=float, default=0.01)
    p.add_argument("--backbone_name", type=str, default=str(defaults.get("backbone_name", "convnext_tiny")))
    p.add_argument("--input_mode", choices=SUPPORTED_STAGE2A_INPUT_MODES, default=str(defaults.get("input_mode", "rgb_mask_geometry")))
    p.add_argument("--pooling_mode", choices=SUPPORTED_STAGE2A_POOLING_MODES, default=str(defaults.get("pooling_mode", "global")))
    p.add_argument("--type_geometry_mode", choices=["none", "concat"], default=str(defaults.get("type_geometry_mode", "concat")))
    p.add_argument("--geometry_cols", type=str, default=",".join(parse_list(defaults.get("geometry_cols"), DEFAULT_GEOMETRY_COLS)))
    p.add_argument("--img_size", type=int, default=int(defaults.get("img_size", 224)))
    p.add_argument("--ring_mask_col", type=str, default=str(defaults.get("ring_mask_col", "mask_R")))
    p.add_argument("--ring_radius_px", type=int, default=int(defaults.get("ring_radius_px", 48)))
    p.add_argument("--num_workers", type=int, default=int(defaults.get("num_workers", 4)))
    p.add_argument("--device", type=str, default=str(defaults.get("device", "cuda")))
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=bool(defaults.get("pretrained", True)))
    p.add_argument("--aug_hflip", type=float, default=float(defaults.get("aug_hflip", 0.5)))
    p.add_argument("--aug_vflip", type=float, default=float(defaults.get("aug_vflip", 0.0)))
    p.add_argument("--aug_rot90", type=float, default=float(defaults.get("aug_rot90", 0.25)))
    p.add_argument("--log_every_steps", type=int, default=int(defaults.get("log_every_steps", 20)))
    return p.parse_args()


def _as_config(args: argparse.Namespace) -> Dict[str, object]:
    cfg = vars(args).copy()
    cfg.update(
        {
            "num_classes": 2,
            "task_mode": "type_only",
            "type_loss_mode": f"m14_specialist_cascade_{args.binary_loss_mode}",
            "type_conditioning_mode": "none",
            "type_geometry_mode": args.type_geometry_mode,
            "geometry_cols": args.geometry_cols,
            "m14_generalist_dir": str(args.generalist_dir),
            "m14_target_classes": args.target_classes,
            "m14_cascade_sources": args.cascade_sources,
        }
    )
    return {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()}


def _autocast_dtype(args: argparse.Namespace, device: torch.device):
    if device.type != "cuda" or args.amp_dtype == "off":
        return None
    return torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16


def _target_indices(target_classes: Sequence[str]) -> List[int]:
    out = []
    for name in target_classes:
        if name not in CLASS_TO_IDX:
            raise ValueError(f"Unknown target class {name!r}; expected one of {CLASS_NAMES}")
        out.append(CLASS_TO_IDX[name])
    return out


def _source(row: Mapping[str, object]) -> str:
    return str(row.get("classification_source", "") or "__missing__")


def make_loader(rows: Sequence[Mapping[str, object]], indices: Sequence[int], args: argparse.Namespace, train: bool, sampler=None) -> DataLoader:
    ds = Stage2aDataset(
        rows,
        indices=indices,
        img_size=args.img_size,
        train=train,
        input_mode=args.input_mode,
        geometry_cols=parse_list(args.geometry_cols, DEFAULT_GEOMETRY_COLS),
        ring_mask_col=args.ring_mask_col,
        ring_radius_px=args.ring_radius_px,
        require_ring_mask=args.pooling_mode == "mask_m_ring" or "ring" in args.input_mode,
        aug_hflip=args.aug_hflip,
        aug_vflip=args.aug_vflip,
        aug_rot90=args.aug_rot90,
        seed=args.seed,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_stage2a,
    )


def make_binary_sampler(rows: Sequence[Mapping[str, object]], indices: Sequence[int], target_idx: int) -> WeightedRandomSampler:
    labels = np.asarray([1 if safe_int(rows[i].get("type_class"), -1) == target_idx else 0 for i in indices], dtype=np.int64)
    pos = max(1, int(labels.sum()))
    neg = max(1, int(len(labels) - labels.sum()))
    weights = np.where(labels == 1, 0.5 / pos, 0.5 / neg)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(indices), replacement=True)


def binary_metrics(y_true: Sequence[int], prob: Sequence[float], threshold: float = 0.5) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(prob, dtype=np.float64)
    pred = (p >= float(threshold)).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    f1 = float((2 * precision * recall) / max(1e-12, precision + recall)) if (precision + recall) > 0 else 0.0
    acc = float((tp + tn) / max(1, len(y)))
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _binary_loss(margin: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(margin, target, pos_weight=pos_weight, reduction="none")
    if args.binary_loss_mode == "focal_bce":
        pt = torch.exp(-bce.detach()).clamp(0.0, 1.0)
        bce = torch.pow(1.0 - pt, float(args.focal_gamma)) * bce
    return bce.mean()


def _save_checkpoint(path: Path, model: torch.nn.Module, args: argparse.Namespace, target_class: str, metrics: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": _as_config(args),
            "target_class": target_class,
            "metrics": dict(metrics),
            "class_names": ["rest", target_class],
        },
        path,
    )


def train_one_specialist(
    rows: Sequence[Mapping[str, object]],
    split: Mapping[str, Sequence[int]],
    target_class: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    target_idx = CLASS_TO_IDX[target_class]
    train_indices = [i for i in split["train"] if _source(rows[i]) == "parcel_code"]
    pos = sum(1 for i in train_indices if safe_int(rows[i].get("type_class"), -1) == target_idx)
    neg = len(train_indices) - pos
    if pos <= 0 or neg <= 0:
        raise ValueError(f"Cannot train {target_class}: parcel_code train positives={pos}, negatives={neg}")

    model = build_model_from_config(_as_config(args), pretrained=args.pretrained).to(device)
    sampler = make_binary_sampler(rows, train_indices, target_idx)
    train_loader = make_loader(rows, train_indices, args, train=True, sampler=sampler)
    val_loader = make_loader(rows, split["val"], args, train=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp_dtype == "fp16" and device.type == "cuda"))
    pos_weight_value = min(float(args.pos_weight_cap), float(neg / max(1, pos)))
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    ac_dtype = _autocast_dtype(args, device)

    target_out = args.out_dir / f"specialist_{target_class}"
    target_out.mkdir(parents=True, exist_ok=True)
    history_path = target_out / "metrics_history.jsonl"
    if history_path.exists():
        history_path.unlink()

    best_score = -1.0
    best_epoch = 0
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        model.train(True)
        total = 0.0
        seen = 0
        for step, batch in enumerate(train_loader, start=1):
            x = batch["x"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            ring_mask = batch["ring_mask"].to(device, non_blocking=True)
            geometry = batch["geometry"].to(device, non_blocking=True)
            y = (batch["type_idx"].to(device, non_blocking=True) == target_idx).float()
            with torch.autocast(device_type=device.type, dtype=ac_dtype, enabled=ac_dtype is not None):
                out = model(x, geometry=geometry, mask=mask, ring_mask=ring_mask, return_dict=True)
                margin = out["type_logits"][:, 1] - out["type_logits"][:, 0]
                loss = _binary_loss(margin, y, pos_weight, args)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            bsz = int(x.shape[0])
            total += float(loss.detach().cpu()) * bsz
            seen += bsz
            if step % max(1, args.log_every_steps) == 0:
                print(f"[m14:{target_class}] epoch={epoch} step={step}/{len(train_loader)} loss={total / max(1, seen):.5f}")

        val_rows = predict_specialist(model, val_loader, target_class, target_idx, device, args)
        parcel_val = [r for r in val_rows if _source(r) == "parcel_code"]
        score = binary_metrics([safe_int(r["binary_target"]) for r in parcel_val], [safe_float(r["specialist_prob"]) for r in parcel_val])["f1"]
        row = {
            "epoch": epoch,
            "train_loss": total / max(1, seen),
            "parcel_code_val_f1_at_05": score,
            "parcel_code_val_support": len(parcel_val),
            "selection_metric": "parcel_code_binary_f1_at_0.5",
            "pos_weight": pos_weight_value,
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"[m14:{target_class}] epoch {epoch} val parcel F1@0.5 {score:.4f}")
        if score > best_score + 1e-12:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            _save_checkpoint(target_out / "best_model.pt", model, args, target_class, row)
            write_csv_rows(target_out / "best_val_predictions.csv", val_rows, specialist_prediction_fields())
        else:
            no_improve += 1
        if no_improve >= args.early_stop_patience:
            print(f"[m14:{target_class}] early_stop no improvement for {no_improve} epochs")
            break

    state = load_torch_checkpoint(target_out / "best_model.pt", map_location=device)
    model.load_state_dict(state["model_state"])
    val_rows = predict_specialist(model, val_loader, target_class, target_idx, device, args)
    test_loader = make_loader(rows, split["test"], args, train=False)
    test_rows = predict_specialist(model, test_loader, target_class, target_idx, device, args)
    write_csv_rows(target_out / "val_predictions.csv", val_rows, specialist_prediction_fields())
    write_csv_rows(target_out / "test_predictions.csv", test_rows, specialist_prediction_fields())
    summary = {
        "target_class": target_class,
        "target_idx": target_idx,
        "train_rows": len(train_indices),
        "train_positive": pos,
        "train_negative": neg,
        "pos_weight": pos_weight_value,
        "best_epoch": best_epoch,
        "best_parcel_code_val_f1_at_05": best_score,
        "val_all_f1_at_05": binary_metrics([safe_int(r["binary_target"]) for r in val_rows], [safe_float(r["specialist_prob"]) for r in val_rows]),
        "val_parcel_code_f1_at_05": binary_metrics(
            [safe_int(r["binary_target"]) for r in val_rows if _source(r) == "parcel_code"],
            [safe_float(r["specialist_prob"]) for r in val_rows if _source(r) == "parcel_code"],
        ),
        "test_parcel_code_f1_at_05_report_only": binary_metrics(
            [safe_int(r["binary_target"]) for r in test_rows if _source(r) == "parcel_code"],
            [safe_float(r["specialist_prob"]) for r in test_rows if _source(r) == "parcel_code"],
        ),
    }
    save_json(target_out / "summary.json", summary)
    return summary


def predict_specialist(
    model: torch.nn.Module,
    loader: DataLoader,
    target_class: str,
    target_idx: int,
    device: torch.device,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    model.eval()
    ac_dtype = _autocast_dtype(args, device)
    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            ring_mask = batch["ring_mask"].to(device, non_blocking=True)
            geometry = batch["geometry"].to(device, non_blocking=True)
            y = (batch["type_idx"].to(device, non_blocking=True) == target_idx).long()
            with torch.autocast(device_type=device.type, dtype=ac_dtype, enabled=ac_dtype is not None):
                out = model(x, geometry=geometry, mask=mask, ring_mask=ring_mask, return_dict=True)
                margin = out["type_logits"][:, 1] - out["type_logits"][:, 0]
                prob = torch.sigmoid(margin).detach().float().cpu().numpy()
            true_type = batch["type_idx"].detach().cpu().numpy()
            for meta, p, yy, tt in zip(batch["meta"], prob, y.detach().cpu().numpy(), true_type):
                rows.append(
                    {
                        "building_uid": meta.get("building_uid", ""),
                        "classification_source": meta.get("classification_source", ""),
                        "GEOID": meta.get("GEOID", ""),
                        "target_class": target_class,
                        "target_idx": target_idx,
                        "specialist_prob": float(p),
                        "binary_target": int(yy),
                        "true_type_idx": int(tt),
                        "true_type_class": CLASS_NAMES[int(tt)] if 0 <= int(tt) < len(CLASS_NAMES) else "",
                    }
                )
    return rows


def specialist_prediction_fields() -> List[str]:
    return [
        "building_uid",
        "classification_source",
        "GEOID",
        "target_class",
        "target_idx",
        "specialist_prob",
        "binary_target",
        "true_type_idx",
        "true_type_class",
    ]


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pred_idx_from_probs(row: Mapping[str, object]) -> int:
    probs = [safe_float(row.get(col), 0.0) for col in PROB_COLUMNS]
    if sum(probs) > 0:
        return int(np.argmax(np.asarray(probs, dtype=np.float64)))
    return safe_int(row.get("pred_type_idx"), -1)


def _metrics_from_pred_rows(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    y_true = [safe_int(r.get("true_type_idx"), -1) for r in rows]
    y_pred = [safe_int(r.get("pred_type_idx"), -1) for r in rows]
    cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < len(CLASS_NAMES) and 0 <= p < len(CLASS_NAMES):
            cm[t, p] += 1
    f1s = []
    for c in range(len(CLASS_NAMES)):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        den = 2 * tp + fp + fn
        f1s.append(float(2 * tp / den) if den > 0 else 0.0)
    acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if y_true else 0.0
    return {
        "type_accuracy": acc,
        "type_macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "type_per_class_f1": {name: f1s[i] for i, name in enumerate(CLASS_NAMES)},
        "confusion_matrix": cm.tolist(),
    }


def _source_slice(rows: Sequence[Mapping[str, object]], target_classes: Sequence[str]) -> Dict[str, object]:
    if not rows:
        return {"support": 0}
    metrics = _metrics_from_pred_rows(rows)
    support = Counter(safe_int(r.get("true_type_idx"), -1) for r in rows)
    preds = Counter(safe_int(r.get("pred_type_idx"), -1) for r in rows)
    support_by_class = {name: int(support.get(i, 0)) for i, name in enumerate(CLASS_NAMES)}
    pred_by_class = {name: int(preds.get(i, 0)) for i, name in enumerate(CLASS_NAMES)}
    present = [name for name, n in support_by_class.items() if n > 0]
    absent = [name for name, n in support_by_class.items() if n == 0]
    per_class = metrics["type_per_class_f1"]
    target_absent_fp: Dict[str, Optional[float]] = {}
    target_fp: Dict[str, float] = {}
    for name in target_classes:
        idx = CLASS_TO_IDX[name]
        non_target = [r for r in rows if safe_int(r.get("true_type_idx"), -1) != idx]
        fp = sum(1 for r in non_target if safe_int(r.get("pred_type_idx"), -1) == idx)
        target_fp[name] = float(fp / max(1, len(non_target)))
        target_absent_fp[name] = float(pred_by_class[name] / max(1, len(rows))) if support_by_class[name] == 0 else None
    return {
        "support": len(rows),
        "support_by_class": support_by_class,
        "prediction_counts": pred_by_class,
        "all_class_macro_f1": metrics["type_macro_f1"],
        "present_class_macro_f1": float(np.mean([safe_float(per_class.get(name), 0.0) for name in present])) if present else 0.0,
        "absent_class_false_positive_rate": float(sum(pred_by_class[name] for name in absent) / max(1, len(rows))),
        "target_absent_class_false_positive_rate": target_absent_fp,
        "target_false_positive_rate": target_fp,
        "per_class_f1": per_class,
    }


def source_summaries_from_pred_rows(rows: Sequence[Mapping[str, object]], target_classes: Sequence[str]) -> Dict[str, object]:
    groups: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[_source(row)].append(row)
    return {source: _source_slice(items, target_classes) for source, items in sorted(groups.items())}


def normalize_generalist_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    out = []
    for row in rows:
        r = dict(row)
        pred = _pred_idx_from_probs(r)
        r["pred_type_idx"] = pred
        r["pred_type_class"] = CLASS_NAMES[pred] if 0 <= pred < len(CLASS_NAMES) else ""
        out.append(r)
    return out


def apply_cascade(
    generalist_rows: Sequence[Mapping[str, object]],
    specialist_probs: Mapping[str, Mapping[str, float]],
    thresholds: Mapping[str, float],
    cascade_sources: Sequence[str],
) -> List[Dict[str, object]]:
    source_set = set(cascade_sources)
    out: List[Dict[str, object]] = []
    for row in normalize_generalist_rows(generalist_rows):
        r = dict(row)
        base_pred = safe_int(r.get("pred_type_idx"), -1)
        chosen = base_pred
        reason = "generalist"
        if _source(r) in source_set:
            uid = str(r.get("building_uid", ""))
            candidates = []
            for target_class, per_uid in specialist_probs.items():
                prob = float(per_uid.get(uid, 0.0))
                thr = float(thresholds[target_class])
                r[f"specialist_prob_{target_class}"] = prob
                candidates.append((prob - thr, prob, CLASS_TO_IDX[target_class], target_class))
            candidates.sort(reverse=True)
            if candidates and candidates[0][0] >= 0.0:
                _, prob, chosen, target_class = candidates[0]
                reason = f"specialist_{target_class}"
                r["pred_type_conf"] = prob
        for col in PROB_COLUMNS:
            r[col] = 0.0
        if 0 <= chosen < len(PROB_COLUMNS):
            r[PROB_COLUMNS[chosen]] = 1.0
        r["pred_type_idx"] = int(chosen)
        r["pred_type_class"] = CLASS_NAMES[int(chosen)] if 0 <= int(chosen) < len(CLASS_NAMES) else ""
        r["cascade_override"] = reason
        out.append(r)
    return out


def _threshold_values(args: argparse.Namespace) -> List[float]:
    vals = []
    cur = float(args.threshold_min)
    while cur <= float(args.threshold_max) + 1e-9:
        vals.append(round(cur, 6))
        cur += float(args.threshold_step)
    return vals


def select_thresholds(
    generalist_val_rows: Sequence[Mapping[str, object]],
    specialist_probs: Mapping[str, Mapping[str, float]],
    target_classes: Sequence[str],
    cascade_sources: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, object]:
    base_rows = normalize_generalist_rows(generalist_val_rows)
    base_metrics = _metrics_from_pred_rows(base_rows)
    base_sources = source_summaries_from_pred_rows(base_rows, target_classes)
    values = _threshold_values(args)
    best_any = None
    best_pass = None
    for combo in product(values, repeat=len(target_classes)):
        thresholds = {name: float(value) for name, value in zip(target_classes, combo)}
        rows = apply_cascade(base_rows, specialist_probs, thresholds, cascade_sources)
        metrics = _metrics_from_pred_rows(rows)
        sources = source_summaries_from_pred_rows(rows, target_classes)
        accuracy_drop = float(base_metrics["type_accuracy"]) - float(metrics["type_accuracy"])
        area_base = base_sources.get("area_heuristic", {})
        area_new = sources.get("area_heuristic", {})
        area_ok = True
        area_increase: Dict[str, float] = {}
        for name in target_classes:
            b = (area_base.get("target_absent_class_false_positive_rate") or {}).get(name) or 0.0
            n = (area_new.get("target_absent_class_false_positive_rate") or {}).get(name) or 0.0
            area_increase[name] = float(n - b)
            if n - b > float(args.max_area_absent_fp_increase):
                area_ok = False
        guardrail_pass = accuracy_drop <= float(args.max_val_accuracy_drop) and area_ok
        candidate = {
            "thresholds": thresholds,
            "val_metrics": metrics,
            "val_source_summaries": sources,
            "base_val_metrics": base_metrics,
            "base_val_source_summaries": base_sources,
            "accuracy_drop": accuracy_drop,
            "area_absent_fp_increase": area_increase,
            "guardrail_pass": guardrail_pass,
            "selection_metric": "validation_macro_f1_with_accuracy_and_area_fp_guardrails",
        }
        if best_any is None or float(metrics["type_macro_f1"]) > float(best_any["val_metrics"]["type_macro_f1"]):
            best_any = candidate
        if guardrail_pass and (best_pass is None or float(metrics["type_macro_f1"]) > float(best_pass["val_metrics"]["type_macro_f1"])):
            best_pass = candidate
    selected = best_pass if best_pass is not None else best_any
    selected["fallback_used"] = best_pass is None
    return selected


def _prob_map(rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    return {str(r.get("building_uid", "")): safe_float(r.get("specialist_prob"), 0.0) for r in rows}


def run_cascade(args: argparse.Namespace, target_classes: Sequence[str]) -> Dict[str, object]:
    val_generalist = read_csv_dicts(args.generalist_dir / "best_val_predictions.csv")
    test_generalist = read_csv_dicts(args.generalist_dir / "test_predictions.csv")
    specialist_val = {}
    specialist_test = {}
    for target_class in target_classes:
        target_dir = args.out_dir / f"specialist_{target_class}"
        specialist_val[target_class] = _prob_map(read_csv_dicts(target_dir / "val_predictions.csv"))
        specialist_test[target_class] = _prob_map(read_csv_dicts(target_dir / "test_predictions.csv"))
    cascade_sources = [x.strip() for x in str(args.cascade_sources).split(",") if x.strip()]
    selected = select_thresholds(val_generalist, specialist_val, target_classes, cascade_sources, args)
    thresholds = selected["thresholds"]
    val_rows = apply_cascade(val_generalist, specialist_val, thresholds, cascade_sources)
    test_rows = apply_cascade(test_generalist, specialist_test, thresholds, cascade_sources)
    write_csv_rows(args.out_dir / "m14_val_predictions.csv", val_rows, cascade_prediction_fields(target_classes))
    write_csv_rows(args.out_dir / "m14_test_predictions.csv", test_rows, cascade_prediction_fields(target_classes))
    write_csv_rows(args.out_dir / "best_val_predictions.csv", val_rows, cascade_prediction_fields(target_classes))
    write_csv_rows(args.out_dir / "test_predictions.csv", test_rows, cascade_prediction_fields(target_classes))
    save_json(args.out_dir / "test_metrics.json", _metrics_from_pred_rows(test_rows))
    return {
        "generalist_dir": str(args.generalist_dir),
        "cascade_sources": cascade_sources,
        "selected_thresholds": thresholds,
        "selection": selected,
        "val_metrics": _metrics_from_pred_rows(val_rows),
        "test_metrics_report_only": _metrics_from_pred_rows(test_rows),
        "val_source_summaries": source_summaries_from_pred_rows(val_rows, target_classes),
        "test_source_summaries_report_only": source_summaries_from_pred_rows(test_rows, target_classes),
        "base_val_metrics": _metrics_from_pred_rows(normalize_generalist_rows(val_generalist)),
        "base_test_metrics_report_only": _metrics_from_pred_rows(normalize_generalist_rows(test_generalist)),
        "base_val_source_summaries": source_summaries_from_pred_rows(normalize_generalist_rows(val_generalist), target_classes),
        "base_test_source_summaries_report_only": source_summaries_from_pred_rows(normalize_generalist_rows(test_generalist), target_classes),
        "protocol_note": "Thresholds are selected on validation only; fixed test metrics are report-only.",
    }


def cascade_prediction_fields(target_classes: Sequence[str]) -> List[str]:
    return [
        "building_uid",
        "pred_population",
        "pred_log1p_population",
        "pred_type_idx",
        "pred_type_class",
        "pred_type_conf",
        *PROB_COLUMNS,
        *[f"specialist_prob_{name}" for name in target_classes],
        "cascade_override",
        "crop_path",
        "mask_path",
        "tile_base",
        "GEOID",
        "classification_source",
        "true_population",
        "true_log1p_population",
        "true_type_idx",
        "true_type_class",
    ]


def _split_summary(rows: Sequence[Mapping[str, object]], split: Mapping[str, Sequence[int]]) -> Dict[str, object]:
    out = {}
    for name, indices in split.items():
        class_counts = Counter(safe_int(rows[i].get("type_class"), -1) for i in indices)
        source_counts = Counter(_source(rows[i]) for i in indices)
        out[name] = {
            "rows": len(indices),
            "class_counts": {CLASS_NAMES[i]: int(class_counts.get(i, 0)) for i in range(len(CLASS_NAMES))},
            "classification_source_counts": dict(source_counts),
        }
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    rows = read_csv_rows(args.labels_csv)
    split = tile_train_val_test_split(rows, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.split_seed, tile_col=args.split_group_col)
    target_classes = [x.strip() for x in str(args.target_classes).split(",") if x.strip()]
    _target_indices(target_classes)
    save_json(args.out_dir / "train_config.json", _as_config(args))
    save_json(args.out_dir / "split_summary.json", _split_summary(rows, split))

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("[info] rows=", len(rows), "split=", {k: len(v) for k, v in split.items()}, "device=", device)
    print("[info] generalist_dir=", args.generalist_dir)

    specialist_summaries = {}
    for target_class in target_classes:
        print(f"TRAIN_SPECIALIST {target_class}")
        specialist_summaries[target_class] = train_one_specialist(rows, split, target_class, args, device)

    cascade_summary = run_cascade(args, target_classes)
    summary = {
        "seed": args.seed,
        "split_seed": args.split_seed,
        "target_classes": target_classes,
        "specialists": specialist_summaries,
        "cascade": cascade_summary,
    }
    save_json(args.out_dir / "m14_cascade_summary.json", summary)
    print("[done] wrote", args.out_dir / "m14_cascade_summary.json")


if __name__ == "__main__":
    main()
