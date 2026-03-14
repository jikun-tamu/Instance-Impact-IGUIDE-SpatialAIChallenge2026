#!/usr/bin/env python3
"""Train Stage-2 Siamese ordinal damage model from preprocessed samples CSV.

Core features:
- Siamese pre/post feature extraction (timm backbone)
- Masked pooling on building mask M and context ring R
- CORAL ordinal head + BCE cumulative loss
- Tile-level train/val split to reduce leakage
- Metrics: macro-F1, QWK, ECE
- AMP, AdamW, checkpointing
- Optional class-balanced sampler
- Optional DDP via torchrun
"""

import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler, WeightedRandomSampler


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args():
    p = argparse.ArgumentParser(description="Train Stage-2 Siamese CORAL model.")
    p.add_argument("--csv", required=True, type=Path, help="Path to stage2_samples.csv")
    p.add_argument("--out_dir", required=True, type=Path, help="Output dir for logs/checkpoints")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--classes", type=int, default=4)
    p.add_argument("--backbone", type=str, default="convnext_tiny")
    p.add_argument("--backbone_stage_index", type=int, default=-1, help="Feature stage index. -1 means last.")
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--crop_size", type=int, default=256, help="Input image size (for info/checks).")
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp_dtype", choices=["bf16", "fp16", "off"], default="bf16")
    p.add_argument("--limit", type=int, default=0, help="Limit CSV rows for quick runs.")
    p.add_argument(
        "--sampler_mode",
        choices=["natural", "weighted", "class_aware"],
        default="natural",
        help="Train sampler mode. 'weighted' matches legacy class_balance behavior.",
    )
    p.add_argument("--class_balance", action="store_true", help="Use weighted random sampler on train split.")
    p.add_argument("--class_balance_alpha", type=float, default=0.5, help="Inverse-frequency weight exponent.")
    p.add_argument("--class_balance_cap", type=float, default=12.0, help="Max per-class weight cap.")
    p.add_argument(
        "--loss_reweight_mode",
        choices=["none", "class_weight", "focal_class_weight"],
        default="none",
        help="Rare-class boosting at loss level (independent of sampler_mode).",
    )
    p.add_argument(
        "--focal_gamma",
        type=float,
        default=1.5,
        help="Focal gamma when loss_reweight_mode=focal_class_weight.",
    )
    p.add_argument(
        "--class_aware_counts",
        type=str,
        default="",
        help="Comma-separated per-class counts per batch for class_aware mode (e.g. 10,4,5,5 for bs=24).",
    )
    p.add_argument(
        "--steps_per_epoch",
        type=int,
        default=0,
        help="Override train steps per epoch (class_aware mode). 0 => auto.",
    )
    p.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use timm pretrained backbone weights (default: on). Use --no-pretrained to disable.",
    )
    p.add_argument("--aug_hflip", type=float, default=0.5, help="Horizontal flip probability.")
    p.add_argument("--aug_vflip", type=float, default=0.0, help="Vertical flip probability.")
    p.add_argument("--aug_rot90", type=float, default=0.5, help="Random 90-degree rotate probability.")
    p.add_argument("--aug_color_jitter", type=float, default=0.2, help="Brightness/contrast jitter strength.")
    p.add_argument(
        "--lr_scheduler",
        choices=["none", "cosine"],
        default="cosine",
        help="Learning rate scheduler.",
    )
    p.add_argument("--warmup_epochs", type=int, default=1, help="Linear warmup epochs before cosine decay.")
    p.add_argument("--log_every_steps", type=int, default=100)
    p.add_argument("--print_per_class_f1", action="store_true", help="Print per-class F1 each validation epoch.")
    p.add_argument("--print_confusion_matrix", action="store_true", help="Print 4x4 validation confusion matrix each epoch.")
    p.add_argument("--ece_bins", type=int, default=15)
    p.add_argument(
        "--best_metric",
        choices=["macro_f1", "qwk", "val_loss"],
        default="macro_f1",
        help="Primary checkpoint selection metric.",
    )
    p.add_argument(
        "--best_tiebreak_metric",
        choices=["none", "qwk", "val_loss", "ece"],
        default="val_loss",
        help="Secondary checkpoint metric when primary ties.",
    )
    p.add_argument(
        "--best_min_delta",
        type=float,
        default=1e-6,
        help="Minimum delta for considering primary metric improvement.",
    )
    p.add_argument(
        "--coral_label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing epsilon for CORAL targets in [0, 0.2).",
    )
    p.add_argument(
        "--ema_decay",
        type=float,
        default=0.0,
        help="EMA decay in [0,1). 0 disables EMA evaluation/checkpointing.",
    )
    p.add_argument(
        "--event_metrics",
        action="store_true",
        help="Log per-event validation macro-F1/QWK if event_id exists in CSV.",
    )
    p.add_argument(
        "--save_val_predictions",
        action="store_true",
        help="Save validation predictions/probabilities for best checkpoint.",
    )
    p.add_argument(
        "--change_fusion",
        choices=["legacy", "pre_post_diff"],
        default="pre_post_diff",
        help="Feature fusion mode before CORAL head.",
    )
    p.add_argument(
        "--pooling_mode",
        choices=["rgb_only", "mask_m", "mask_m_ring"],
        default="mask_m_ring",
        help="Ablation mode for pooled features: RGB global only, mask M only, or mask M + ring R.",
    )
    p.add_argument(
        "--diff_abs_scale",
        type=float,
        default=1.0,
        help="Scale multiplier applied to absolute diff branch in pre_post_diff fusion.",
    )
    p.add_argument(
        "--hard_example_mining",
        action="store_true",
        help="Enable online hard-example reweighting (sampler_mode=natural only).",
    )
    p.add_argument(
        "--hard_mining_alpha",
        type=float,
        default=0.5,
        help="Mixing coefficient between uniform and hardness weights.",
    )
    p.add_argument(
        "--hard_mining_ema",
        type=float,
        default=0.8,
        help="EMA coefficient for per-sample hardness memory.",
    )
    p.add_argument(
        "--hard_mining_warmup_epochs",
        type=int,
        default=1,
        help="Number of epochs before hard-example mining starts.",
    )
    p.add_argument("--save_every_epoch", action="store_true")
    p.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Early stopping patience on val macro-F1. 0 disables early stopping.",
    )
    return p.parse_args()


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def is_main():
    return get_rank() == 0


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, None
    if not torch.cuda.is_available():
        raise RuntimeError("DDP requested (WORLD_SIZE>1), but CUDA is unavailable.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, dist.get_rank(), dist.get_world_size(), local_rank


def cleanup_ddp():
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_rows(csv_path, limit=0):
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit > 0 and i >= limit:
                break
            rows.append(row)
    return rows


def tile_split_indices(rows, val_ratio, seed):
    tile_ids = sorted({r["tile_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(tile_ids)
    n_val_tiles = max(1, int(round(len(tile_ids) * val_ratio)))
    val_tiles = set(tile_ids[:n_val_tiles])
    train_idx, val_idx = [], []
    for i, r in enumerate(rows):
        if r["tile_id"] in val_tiles:
            val_idx.append(i)
        else:
            train_idx.append(i)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("Invalid split produced empty train or val set.")
    return train_idx, val_idx


def load_rgb_tensor(path):
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # HWC -> CHW
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t


def load_mask_tensor(path):
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    m = (arr > 0).astype(np.float32)
    return torch.from_numpy(m).unsqueeze(0).contiguous()  # [1,H,W]


def apply_pair_augment(pre, post, m, r, cfg, rng):
    # Geometric transforms must stay synchronized across pre/post/masks.
    if rng.random() < cfg["hflip"]:
        pre = torch.flip(pre, dims=[2])
        post = torch.flip(post, dims=[2])
        m = torch.flip(m, dims=[2])
        r = torch.flip(r, dims=[2])
    if rng.random() < cfg["vflip"]:
        pre = torch.flip(pre, dims=[1])
        post = torch.flip(post, dims=[1])
        m = torch.flip(m, dims=[1])
        r = torch.flip(r, dims=[1])
    if rng.random() < cfg["rot90"]:
        k = rng.randint(1, 3)
        pre = torch.rot90(pre, k=k, dims=[1, 2])
        post = torch.rot90(post, k=k, dims=[1, 2])
        m = torch.rot90(m, k=k, dims=[1, 2])
        r = torch.rot90(r, k=k, dims=[1, 2])

    # Mild color jitter, independently per image.
    cj = cfg["color_jitter"]
    if cj > 0:
        b1 = 1.0 + rng.uniform(-cj, cj)
        c1 = 1.0 + rng.uniform(-cj, cj)
        b2 = 1.0 + rng.uniform(-cj, cj)
        c2 = 1.0 + rng.uniform(-cj, cj)
        pre = ((pre * c1) + (b1 - 1.0)).clamp(-5.0, 5.0)
        post = ((post * c2) + (b2 - 1.0)).clamp(-5.0, 5.0)

    return pre.contiguous(), post.contiguous(), m.contiguous(), r.contiguous()


class Stage2Dataset(Dataset):
    def __init__(self, rows, indices, train=False, aug_cfg=None, seed=42):
        self.rows = rows
        self.indices = indices
        self.train = train
        self.aug_cfg = aug_cfg or {}
        self.seed = seed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.rows[self.indices[i]]
        pre = load_rgb_tensor(row["pre_crop"])
        post = load_rgb_tensor(row["post_crop"])
        m = load_mask_tensor(row["mask_M"])
        r = load_mask_tensor(row["mask_R"])
        if self.train and self.aug_cfg:
            rng = random.Random((self.seed * 1_000_003) + i)
            pre, post, m, r = apply_pair_augment(pre, post, m, r, self.aug_cfg, rng)
        y = int(row["damage_class"])
        meta = {"tile_id": row.get("tile_id", ""), "event_id": row.get("event_id", ""), "bldg_uid": row.get("bldg_uid", "")}
        return pre, post, m, r, y, meta, i


class DistributedWeightedSampler(Sampler):
    """Distributed weighted sampling with replacement for imbalanced DDP training."""

    def __init__(self, weights, num_replicas=None, rank=None, seed=42):
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.weights) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        sampled = torch.multinomial(
            self.weights,
            num_samples=self.total_size,
            replacement=True,
            generator=g,
        ).tolist()
        indices = sampled[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_weights(self, weights):
        if len(weights) != len(self.weights):
            raise ValueError("weights length mismatch in DistributedWeightedSampler.set_weights")
        self.weights = torch.as_tensor(weights, dtype=torch.double)


class ClassAwareBatchSampler(Sampler):
    """Build each batch with explicit per-class counts."""

    def __init__(self, class_buckets, batch_counts, steps_per_epoch, seed=42):
        self.class_buckets = [list(x) for x in class_buckets]
        self.batch_counts = list(batch_counts)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.epoch = 0
        self.num_classes = len(self.class_buckets)
        if len(self.batch_counts) != self.num_classes:
            raise ValueError("batch_counts length must match number of classes")
        if any(c < 0 for c in self.batch_counts):
            raise ValueError("batch_counts must be >= 0")
        if sum(self.batch_counts) <= 0:
            raise ValueError("sum(batch_counts) must be > 0")
        for c, bucket in enumerate(self.class_buckets):
            if self.batch_counts[c] > 0 and len(bucket) == 0:
                raise ValueError(f"class {c} has zero samples but requires {self.batch_counts[c]} per batch")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        pools = [list(b) for b in self.class_buckets]
        ptrs = [0 for _ in pools]
        for b in pools:
            rng.shuffle(b)

        def draw_from_class(cls_id, n):
            out = []
            pool = pools[cls_id]
            while len(out) < n:
                if ptrs[cls_id] >= len(pool):
                    rng.shuffle(pool)
                    ptrs[cls_id] = 0
                out.append(pool[ptrs[cls_id]])
                ptrs[cls_id] += 1
            return out

        for _ in range(self.steps_per_epoch):
            batch = []
            for c in range(self.num_classes):
                n = self.batch_counts[c]
                if n > 0:
                    batch.extend(draw_from_class(c, n))
            rng.shuffle(batch)
            yield batch


def collate_batch(batch):
    pre, post, m, r, y, meta, sample_id = zip(*batch)
    return (
        torch.stack(pre, dim=0),
        torch.stack(post, dim=0),
        torch.stack(m, dim=0),
        torch.stack(r, dim=0),
        torch.tensor(y, dtype=torch.long),
        meta,
        torch.tensor(sample_id, dtype=torch.long),
    )


def downsample_mask(mask, h, w):
    return F.interpolate(mask, size=(h, w), mode="nearest")


def masked_avg_pool(feat, mask, eps=1e-6):
    # feat: [B,C,H,W], mask: [B,1,Hm,Wm] in {0,1}
    m = downsample_mask(mask, feat.shape[-2], feat.shape[-1])
    num = (feat * m).sum(dim=(2, 3))
    den = m.sum(dim=(2, 3)).clamp_min(eps)
    return num / den


def global_avg_pool(feat):
    return feat.mean(dim=(2, 3))


class CoralHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes - 1),
        )

    def forward(self, x):
        return self.net(x)


def coral_targets(y, num_classes, label_smoothing=0.0):
    # target[k] = 1 if y > k else 0 for k in [0..K-2]
    ks = torch.arange(num_classes - 1, device=y.device).view(1, -1)
    tgt = (y.view(-1, 1) > ks).float()
    if label_smoothing > 0:
        eps = float(label_smoothing)
        tgt = tgt * (1.0 - eps) + 0.5 * eps
    return tgt


def coral_probs_from_logits(logits):
    # logits represent P(y > k) via sigmoid
    p_gt = torch.sigmoid(logits)  # [B,K-1]
    b, km1 = p_gt.shape
    k = km1 + 1
    probs = []
    probs.append(1.0 - p_gt[:, 0])
    for c in range(1, k - 1):
        probs.append(p_gt[:, c - 1] - p_gt[:, c])
    probs.append(p_gt[:, -1])
    probs = torch.stack(probs, dim=1).clamp_min(1e-8)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return probs


class SiameseDamageModel(nn.Module):
    def __init__(
        self,
        backbone_name,
        num_classes,
        hidden_dim=512,
        dropout=0.1,
        pretrained=False,
        stage_index=-1,
        change_fusion="pre_post_diff",
        diff_abs_scale=1.0,
        pooling_mode="mask_m_ring",
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
        )
        channels = self.backbone.feature_info.channels()
        if stage_index < 0:
            stage_index = len(channels) + stage_index
        if stage_index < 0 or stage_index >= len(channels):
            raise ValueError(f"Invalid stage_index {stage_index}, available [0..{len(channels)-1}]")
        self.stage_index = stage_index
        self.change_fusion = change_fusion
        self.diff_abs_scale = float(diff_abs_scale)
        self.pooling_mode = pooling_mode
        c = channels[stage_index]
        if pooling_mode not in ("rgb_only", "mask_m", "mask_m_ring"):
            raise ValueError(f"Unknown pooling_mode={pooling_mode}")
        if change_fusion not in ("legacy", "pre_post_diff"):
            raise ValueError(f"Unknown change_fusion={change_fusion}")
        if pooling_mode == "mask_m_ring":
            in_dim = c * (6 if change_fusion == "legacy" else 8)
        else:
            in_dim = c * (3 if change_fusion == "legacy" else 4)
        self.head = CoralHead(in_dim=in_dim, num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout)

    def feat_map(self, x):
        feats = self.backbone(x)
        return feats[self.stage_index]

    def forward(self, pre, post, m, r):
        f_pre = self.feat_map(pre)
        f_post = self.feat_map(post)
        if self.pooling_mode == "rgb_only":
            v_pre = global_avg_pool(f_pre)
            v_post = global_avg_pool(f_post)
            d = v_post - v_pre
            if self.change_fusion == "legacy":
                x = torch.cat([v_pre, v_post, d], dim=1)
            else:
                d_abs = self.diff_abs_scale * torch.abs(d)
                x = torch.cat([v_pre, v_post, d, d_abs], dim=1)
        elif self.pooling_mode == "mask_m":
            v_pre = masked_avg_pool(f_pre, m)
            v_post = masked_avg_pool(f_post, m)
            d = v_post - v_pre
            if self.change_fusion == "legacy":
                x = torch.cat([v_pre, v_post, d], dim=1)
            else:
                d_abs = self.diff_abs_scale * torch.abs(d)
                x = torch.cat([v_pre, v_post, d, d_abs], dim=1)
        else:
            v_pre_m = masked_avg_pool(f_pre, m)
            v_post_m = masked_avg_pool(f_post, m)
            v_pre_r = masked_avg_pool(f_pre, r)
            v_post_r = masked_avg_pool(f_post, r)
            d_m = v_post_m - v_pre_m
            d_r = v_post_r - v_pre_r
            if self.change_fusion == "legacy":
                x = torch.cat([v_pre_m, v_post_m, v_pre_r, v_post_r, d_m, d_r], dim=1)
            else:
                d_abs_m = self.diff_abs_scale * torch.abs(d_m)
                d_abs_r = self.diff_abs_scale * torch.abs(d_r)
                x = torch.cat([v_pre_m, v_post_m, v_pre_r, v_post_r, d_m, d_r, d_abs_m, d_abs_r], dim=1)
        logits = self.head(x)
        probs = coral_probs_from_logits(logits)
        return {"logits_cum": logits, "probs": probs}


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def macro_f1_from_cm(cm):
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        den = (2 * tp + fp + fn)
        f1 = (2 * tp / den) if den > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)), [float(x) for x in f1s]


def qwk_from_cm(cm):
    n = cm.sum()
    if n == 0:
        return 0.0
    k = cm.shape[0]
    w = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            w[i, j] = ((i - j) ** 2) / ((k - 1) ** 2)
    act = cm.sum(axis=1)
    pred = cm.sum(axis=0)
    e = np.outer(act, pred) / max(1, n)
    num = (w * cm).sum()
    den = (w * e).sum()
    if den <= 0:
        return 0.0
    return float(1.0 - num / den)


def ece_score(probs, y_true, n_bins=15):
    # probs: [N,K], y_true: [N]
    if len(y_true) == 0:
        return 0.0
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (conf >= lo) & (conf <= hi)
        else:
            m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            continue
        acc_bin = acc[m].mean()
        conf_bin = conf[m].mean()
        ece += float(m.sum()) / n * abs(float(acc_bin) - float(conf_bin))
    return float(ece)


def gather_list_ddp(local_list):
    if not is_dist():
        return local_list
    gathered = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, local_list)
    merged = []
    for g in gathered:
        merged.extend(g)
    return merged


def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    num_classes,
    amp_dtype,
    train=True,
    log_every_steps=100,
    class_weight_tensor=None,
    loss_reweight_mode="none",
    focal_gamma=1.5,
    label_smoothing=0.0,
    ece_bins=15,
    collect_outputs=False,
    collect_hardness=False,
    ema_state=None,
):
    model.train(train)
    total_loss = 0.0
    n_seen = 0
    ys, ps, probs_all = [], [], []
    event_ids = []
    hard_pairs = []
    t0 = time.time()

    if amp_dtype == "bf16":
        ac_dtype = torch.bfloat16
    elif amp_dtype == "fp16":
        ac_dtype = torch.float16
    else:
        ac_dtype = None

    for step, batch in enumerate(loader, start=1):
        pre, post, m, r, y, meta, sample_id = batch
        pre = pre.to(device, non_blocking=True)
        post = post.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        r = r.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        tgt = coral_targets(y, num_classes, label_smoothing=label_smoothing if train else 0.0)

        if train:
            optimizer.zero_grad(set_to_none=True)

        use_amp = ac_dtype is not None and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=ac_dtype, enabled=use_amp):
            out = model(pre, post, m, r)
            bce_raw = F.binary_cross_entropy_with_logits(out["logits_cum"], tgt, reduction="none")  # [B, K-1]
            sample_loss = bce_raw.mean(dim=1)  # [B]
            if loss_reweight_mode == "none":
                loss = sample_loss.mean()
            else:
                # CORAL element-wise BCE first, then optional focal + class-weight scaling.
                sample_weight = class_weight_tensor[y].float() if class_weight_tensor is not None else torch.ones_like(sample_loss)

                if loss_reweight_mode == "class_weight":
                    loss = (sample_loss * sample_weight).mean()
                elif loss_reweight_mode == "focal_class_weight":
                    p_t = torch.exp(-sample_loss).clamp_min(1e-8)  # p(correct)
                    focal = (1.0 - p_t) ** float(focal_gamma)
                    loss = (sample_loss * focal * sample_weight).mean()
                else:
                    raise ValueError(f"Unknown loss_reweight_mode={loss_reweight_mode}")

        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if ema_state is not None:
                ema_state.update(model)

        bsz = y.shape[0]
        total_loss += float(loss.detach().cpu().item()) * bsz
        n_seen += bsz
        p = out["probs"].detach().float().cpu().numpy()
        pred = p.argmax(axis=1).tolist()
        ys.extend(y.detach().cpu().tolist())
        ps.extend(pred)
        probs_all.extend(p.tolist())
        event_ids.extend([x.get("event_id", "") for x in meta])
        if collect_hardness and train:
            sid = sample_id.detach().cpu().tolist()
            sl = sample_loss.detach().float().cpu().tolist()
            hard_pairs.extend(list(zip(sid, sl)))

        if is_main() and log_every_steps > 0 and step % log_every_steps == 0:
            elapsed = time.time() - t0
            print(
                ("train" if train else "val"),
                "step",
                step,
                "| loss",
                f"{(total_loss / max(1, n_seen)):.5f}",
                "| seen",
                n_seen,
                "| elapsed_s",
                f"{elapsed:.1f}",
            )

    # Correct DDP loss aggregation: all-reduce local sums/counts before averaging.
    loss_sum_t = torch.tensor([total_loss], dtype=torch.float64, device=device)
    seen_t = torch.tensor([n_seen], dtype=torch.float64, device=device)
    if is_dist():
        dist.all_reduce(loss_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(seen_t, op=dist.ReduceOp.SUM)
    loss_avg = float((loss_sum_t / seen_t.clamp_min(1.0)).item())

    ys = gather_list_ddp(ys)
    ps = gather_list_ddp(ps)
    probs_all = gather_list_ddp(probs_all)
    event_ids = gather_list_ddp(event_ids)
    hard_pairs = gather_list_ddp(hard_pairs)
    n_seen = len(ys)
    if n_seen == 0:
        out = {"loss": 0.0, "macro_f1": 0.0, "qwk": 0.0, "ece": 0.0, "per_class_f1": [0.0] * num_classes}
        out["confusion_matrix"] = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
        if collect_outputs:
            out["ys"] = []
            out["ps"] = []
            out["probs"] = []
            out["event_ids"] = []
        if collect_hardness and train:
            out["hard_pairs"] = []
        return out

    probs_np = np.array(probs_all, dtype=np.float32)
    ys_np = np.array(ys, dtype=np.int64)
    ps_np = np.array(ps, dtype=np.int64)
    cm = confusion_matrix(ys_np, ps_np, num_classes)
    macro_f1, per_class_f1 = macro_f1_from_cm(cm)
    qwk = qwk_from_cm(cm)
    ece = ece_score(probs_np, ys_np, n_bins=ece_bins)
    out = {"loss": loss_avg, "macro_f1": macro_f1, "qwk": qwk, "ece": ece, "per_class_f1": per_class_f1}
    out["confusion_matrix"] = cm.tolist()
    if collect_outputs:
        out["ys"] = ys
        out["ps"] = ps
        out["probs"] = probs_all
        out["event_ids"] = event_ids
    if collect_hardness and train:
        out["hard_pairs"] = hard_pairs
    return out


def per_event_metrics(ys, ps, event_ids, num_classes):
    buckets = {}
    for y, p, e in zip(ys, ps, event_ids):
        key = e if e else "unknown"
        buckets.setdefault(key, {"y": [], "p": []})
        buckets[key]["y"].append(int(y))
        buckets[key]["p"].append(int(p))
    rows = []
    for event_id, d in sorted(buckets.items(), key=lambda x: x[0]):
        ys_np = np.asarray(d["y"], dtype=np.int64)
        ps_np = np.asarray(d["p"], dtype=np.int64)
        cm = confusion_matrix(ys_np, ps_np, num_classes)
        macro_f1, _ = macro_f1_from_cm(cm)
        qwk = qwk_from_cm(cm)
        rows.append({"event_id": event_id, "n": int(len(ys_np)), "macro_f1": float(macro_f1), "qwk": float(qwk)})
    return rows


def make_train_sampler(rows, train_idx):
    labels = [int(rows[i]["damage_class"]) for i in train_idx]
    cnt = Counter(labels)
    weights = [1.0 / max(1, cnt[y]) for y in labels]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def build_class_buckets(rows, train_idx, num_classes):
    buckets = [[] for _ in range(num_classes)]
    # dataset-local indices are [0..len(train_idx)-1]
    for local_i, row_i in enumerate(train_idx):
        y = int(rows[row_i]["damage_class"])
        if 0 <= y < num_classes:
            buckets[y].append(local_i)
    return buckets


def shard_class_buckets_for_rank(class_buckets, rank, world):
    local = []
    for b in class_buckets:
        if world <= 1:
            local_b = list(b)
        else:
            local_b = b[rank::world]
            # Prevent empty local minority bucket in DDP; fallback to global bucket.
            if len(local_b) == 0 and len(b) > 0:
                local_b = list(b)
        local.append(local_b)
    return local


def resolve_class_aware_counts(spec, batch_size, num_classes, class_counts):
    if spec.strip():
        vals = [int(x.strip()) for x in spec.split(",") if x.strip()]
        if len(vals) != num_classes:
            raise ValueError(f"class_aware_counts must have {num_classes} entries")
        if sum(vals) != batch_size:
            raise ValueError("sum(class_aware_counts) must equal batch_size")
        return vals

    # Auto mode: equal base + remainder assigned to rarer classes first.
    base = batch_size // num_classes
    rem = batch_size - base * num_classes
    counts = [base for _ in range(num_classes)]
    rare_order = sorted(range(num_classes), key=lambda c: class_counts.get(c, 0))
    for i in range(rem):
        counts[rare_order[i % num_classes]] += 1
    return counts


def make_weights_for_indices(rows, train_idx, alpha=0.5, cap=12.0):
    labels = [int(rows[i]["damage_class"]) for i in train_idx]
    cnt = Counter(labels)
    max_count = max(cnt.values()) if cnt else 1
    weights = []
    for y in labels:
        wy = (max_count / max(1, cnt[y])) ** alpha
        weights.append(min(float(wy), float(cap)))
    return weights


def compute_class_weights(rows, train_idx, num_classes, alpha=0.5, cap=12.0):
    labels = [int(rows[i]["damage_class"]) for i in train_idx]
    cnt = Counter(labels)
    max_count = max(cnt.values()) if cnt else 1
    w = []
    for c in range(num_classes):
        wc = (max_count / max(1, cnt.get(c, 0))) ** alpha
        wc = min(float(wc), float(cap))
        w.append(wc)
    return cnt, w


def save_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def to_jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    return obj


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


class EMAState:
    def __init__(self, model, decay):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        base = unwrap_model(model)
        for k, v in base.state_dict().items():
            if torch.is_floating_point(v):
                self.shadow[k] = v.detach().clone()

    def update(self, model):
        base = unwrap_model(model)
        for k, v in base.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model):
        base = unwrap_model(model)
        self.backup = {}
        sd = base.state_dict()
        for k, ema_v in self.shadow.items():
            if k in sd:
                self.backup[k] = sd[k].detach().clone()
                sd[k].copy_(ema_v)

    def restore(self, model):
        if not self.backup:
            return
        base = unwrap_model(model)
        sd = base.state_dict()
        for k, old_v in self.backup.items():
            if k in sd:
                sd[k].copy_(old_v)
        self.backup = {}


def metric_value(metrics, name):
    if name == "macro_f1":
        return float(metrics["macro_f1"])
    if name == "qwk":
        return float(metrics["qwk"])
    if name == "val_loss":
        return float(metrics["loss"])
    if name == "ece":
        return float(metrics["ece"])
    raise ValueError(f"Unknown metric name: {name}")


def is_better_metrics(candidate, best, primary_name, tie_name, min_delta):
    if best is None:
        return True
    c_primary = metric_value(candidate, primary_name)
    b_primary = metric_value(best, primary_name)
    if primary_name == "val_loss":
        if c_primary < (b_primary - min_delta):
            return True
        if c_primary > (b_primary + min_delta):
            return False
    else:
        if c_primary > (b_primary + min_delta):
            return True
        if c_primary < (b_primary - min_delta):
            return False
    if tie_name == "none":
        return False
    c_tie = metric_value(candidate, tie_name)
    b_tie = metric_value(best, tie_name)
    if tie_name in ("val_loss", "ece"):
        return c_tie < b_tie
    return c_tie > b_tie


def main():
    args = parse_args()
    if not (0.0 <= args.coral_label_smoothing < 0.2):
        raise ValueError("--coral_label_smoothing must be in [0, 0.2)")
    if not (0.0 <= args.ema_decay < 1.0):
        raise ValueError("--ema_decay must be in [0, 1)")
    if args.hard_example_mining and args.sampler_mode != "natural":
        raise ValueError("--hard_example_mining currently supports --sampler_mode natural only")
    ddp, rank, world, local_rank = setup_ddp()
    try:
        seed_all(args.seed + rank)

        device = torch.device(f"cuda:{local_rank}" if ddp else ("cuda" if torch.cuda.is_available() else "cpu"))
        if is_main():
            print("Device:", device, "| DDP:", ddp, "| world_size:", world)
            print("Loading rows from", args.csv)

        rows = read_rows(args.csv, limit=args.limit)
        if len(rows) == 0:
            raise RuntimeError("No rows loaded from CSV.")
        train_idx, val_idx = tile_split_indices(rows, val_ratio=args.val_ratio, seed=args.seed)
        # Backward compatibility: old flag maps to weighted mode when sampler_mode isn't explicitly changed.
        if args.class_balance and args.sampler_mode == "natural":
            args.sampler_mode = "weighted"

        if is_main():
            print("Rows:", len(rows), "| train:", len(train_idx), "| val:", len(val_idx))

        train_class_counts, train_class_weights = compute_class_weights(
            rows,
            train_idx,
            num_classes=args.classes,
            alpha=args.class_balance_alpha,
            cap=args.class_balance_cap,
        )
        if is_main():
            print("Train class counts:", dict(sorted(train_class_counts.items())))
            if args.sampler_mode == "weighted":
                print("Class balance enabled. Class weights:", [round(x, 4) for x in train_class_weights])
            if args.loss_reweight_mode != "none":
                print(
                    "Loss reweight enabled:",
                    args.loss_reweight_mode,
                    "| class weights:",
                    [round(x, 4) for x in train_class_weights],
                    "| focal_gamma:",
                    args.focal_gamma,
                )

        aug_cfg = {
            "hflip": args.aug_hflip,
            "vflip": args.aug_vflip,
            "rot90": args.aug_rot90,
            "color_jitter": args.aug_color_jitter,
        }
        train_ds = Stage2Dataset(rows, train_idx, train=True, aug_cfg=aug_cfg, seed=args.seed + rank)
        val_ds = Stage2Dataset(rows, val_idx, train=False, aug_cfg=None, seed=args.seed + rank)

        train_sampler = None
        train_batch_sampler = None
        val_sampler = None
        shuffle_train = True
        train_weights = None

        if args.sampler_mode == "class_aware":
            class_buckets = build_class_buckets(rows, train_idx, args.classes)
            batch_counts = resolve_class_aware_counts(
                args.class_aware_counts,
                batch_size=args.batch_size,
                num_classes=args.classes,
                class_counts=train_class_counts,
            )
            local_buckets = shard_class_buckets_for_rank(class_buckets, rank=rank, world=world if ddp else 1)
            local_train_n = sum(len(b) for b in local_buckets)
            if args.steps_per_epoch > 0:
                steps_per_epoch = args.steps_per_epoch
            else:
                # In DDP, ranks may have slightly different local bucket sizes after sharding.
                # Use a synchronized max so every rank runs identical step counts and avoids deadlock.
                ref_train_n = local_train_n
                if ddp:
                    t_n = torch.tensor([local_train_n], dtype=torch.int64, device=device)
                    dist.all_reduce(t_n, op=dist.ReduceOp.MAX)
                    ref_train_n = int(t_n.item())
                steps_per_epoch = max(1, int(math.ceil(ref_train_n / max(1, args.batch_size))))
            train_batch_sampler = ClassAwareBatchSampler(
                class_buckets=local_buckets,
                batch_counts=batch_counts,
                steps_per_epoch=steps_per_epoch,
                seed=args.seed,
            )
            if is_main():
                print("Class-aware sampler enabled.")
                print("Class-aware batch counts:", batch_counts, "| steps_per_epoch:", steps_per_epoch)
            if ddp:
                val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
        elif ddp:
            if args.sampler_mode == "weighted":
                train_weights = make_weights_for_indices(
                    rows,
                    train_idx,
                    alpha=args.class_balance_alpha,
                    cap=args.class_balance_cap,
                )
                train_sampler = DistributedWeightedSampler(
                    weights=train_weights,
                    num_replicas=world,
                    rank=rank,
                    seed=args.seed,
                )
            elif args.hard_example_mining:
                train_weights = [1.0 for _ in range(len(train_idx))]
                train_sampler = DistributedWeightedSampler(
                    weights=train_weights,
                    num_replicas=world,
                    rank=rank,
                    seed=args.seed,
                )
            else:
                train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
            val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
            shuffle_train = False
        elif args.sampler_mode == "weighted":
            train_sampler = make_train_sampler(rows, train_idx)
            shuffle_train = False
        elif args.hard_example_mining:
            train_weights = [1.0 for _ in range(len(train_idx))]
            train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)
            shuffle_train = False

        if train_batch_sampler is not None:
            train_loader = DataLoader(
                train_ds,
                batch_sampler=train_batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=args.num_workers > 0,
                collate_fn=collate_batch,
            )
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=shuffle_train,
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=args.num_workers > 0,
                collate_fn=collate_batch,
                drop_last=False,
            )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            collate_fn=collate_batch,
            drop_last=False,
        )

        model = SiameseDamageModel(
            backbone_name=args.backbone,
            num_classes=args.classes,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            pretrained=args.pretrained,
            stage_index=args.backbone_stage_index,
            change_fusion=args.change_fusion,
            diff_abs_scale=args.diff_abs_scale,
            pooling_mode=args.pooling_mode,
        ).to(device)
        if ddp:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        class_weight_tensor = None
        if args.loss_reweight_mode != "none":
            class_weight_tensor = torch.tensor(train_class_weights, dtype=torch.float32, device=device)
        scheduler = None
        if args.lr_scheduler == "cosine":
            total_epochs = max(1, args.epochs)
            warmup_epochs = max(0, min(args.warmup_epochs, total_epochs - 1))

            def lr_lambda(epoch_idx):
                if warmup_epochs > 0 and epoch_idx < warmup_epochs:
                    return float(epoch_idx + 1) / float(max(1, warmup_epochs))
                progress = (epoch_idx - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
                progress = min(max(progress, 0.0), 1.0)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        use_fp16_scaler = args.amp_dtype == "fp16" and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)

        args_json = to_jsonable(vars(args))
        if is_main():
            args.out_dir.mkdir(parents=True, exist_ok=True)
            save_json(args.out_dir / "train_config.json", args_json)

        ema_state = EMAState(model, args.ema_decay) if args.ema_decay > 0 else None
        best_metrics = None
        best_epoch = 0
        no_improve_epochs = 0
        hard_scores = np.ones(len(train_idx), dtype=np.float32) if args.hard_example_mining else None
        history = []
        t_train = time.time()
        for epoch in range(1, args.epochs + 1):
            if train_batch_sampler is not None:
                train_batch_sampler.set_epoch(epoch)
            elif ddp:
                train_sampler.set_epoch(epoch)
            if is_main():
                print(f"\nEpoch {epoch}/{args.epochs}")

            tr = run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                num_classes=args.classes,
                amp_dtype=args.amp_dtype,
                train=True,
                log_every_steps=args.log_every_steps,
                class_weight_tensor=class_weight_tensor,
                loss_reweight_mode=args.loss_reweight_mode,
                focal_gamma=args.focal_gamma,
                label_smoothing=args.coral_label_smoothing,
                ece_bins=args.ece_bins,
                collect_hardness=args.hard_example_mining,
                ema_state=ema_state,
            )

            use_ema_eval = ema_state is not None and epoch > max(1, args.warmup_epochs)
            if use_ema_eval:
                ema_state.apply_to(model)
            va = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                scaler=None,
                device=device,
                num_classes=args.classes,
                amp_dtype=args.amp_dtype,
                train=False,
                log_every_steps=args.log_every_steps,
                class_weight_tensor=None,
                loss_reweight_mode="none",
                focal_gamma=args.focal_gamma,
                label_smoothing=0.0,
                ece_bins=args.ece_bins,
                collect_outputs=args.event_metrics or args.save_val_predictions,
            )
            if use_ema_eval:
                ema_state.restore(model)
            if scheduler is not None:
                scheduler.step()

            if args.hard_example_mining and epoch >= args.hard_mining_warmup_epochs:
                epoch_losses = {}
                for sample_id, sample_loss in tr.get("hard_pairs", []):
                    epoch_losses.setdefault(int(sample_id), []).append(float(sample_loss))
                if epoch_losses:
                    cur = hard_scores.copy()
                    for sid, vals in epoch_losses.items():
                        cur[sid] = float(np.mean(vals))
                    hard_scores = (args.hard_mining_ema * hard_scores) + ((1.0 - args.hard_mining_ema) * cur)
                    hs = hard_scores / max(1e-8, float(np.mean(hard_scores)))
                    mixed = (1.0 - args.hard_mining_alpha) + args.hard_mining_alpha * hs
                    mixed = np.clip(mixed, 1e-3, 100.0)
                    if isinstance(train_sampler, DistributedWeightedSampler):
                        train_sampler.set_weights(mixed.tolist())
                    elif isinstance(train_sampler, WeightedRandomSampler):
                        train_sampler.weights = torch.as_tensor(mixed, dtype=torch.double)

            epoch_row = {
                "epoch": epoch,
                "train_loss": tr["loss"],
                "val_loss": va["loss"],
                "macro_f1": va["macro_f1"],
                "qwk": va["qwk"],
                "ece": va["ece"],
                "per_class_f1": va["per_class_f1"],
                "lr": optimizer.param_groups[0]["lr"],
                "eval_mode": "ema" if use_ema_eval else "raw",
            }
            history.append(epoch_row)

            if is_main():
                print(
                    "epoch",
                    epoch,
                    "| train_loss",
                    f"{tr['loss']:.5f}",
                    "| val_loss",
                    f"{va['loss']:.5f}",
                    "| macro_f1",
                    f"{va['macro_f1']:.4f}",
                    "| qwk",
                    f"{va['qwk']:.4f}",
                    "| ece",
                    f"{va['ece']:.4f}",
                    "| lr",
                    f"{optimizer.param_groups[0]['lr']:.3e}",
                    "| eval",
                    epoch_row["eval_mode"],
                )
                if args.print_per_class_f1:
                    f1s = va.get("per_class_f1", [])
                    f1_msg = " ".join([f"c{i}:{float(v):.4f}" for i, v in enumerate(f1s)])
                    print("val per_class_f1 |", f1_msg)
                if args.print_confusion_matrix:
                    cm = va.get("confusion_matrix", [])
                    print("val confusion_matrix:")
                    for row_cm in cm:
                        print(" ", " ".join([str(int(x)) for x in row_cm]))

                ckpt_last = {
                    "epoch": epoch,
                    "model_state": unwrap_model(model).state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "args": args_json,
                    "metrics": epoch_row,
                }
                torch.save(ckpt_last, args.out_dir / "stage2_last.pt")
                if args.save_every_epoch:
                    torch.save(ckpt_last, args.out_dir / f"stage2_epoch{epoch:02d}.pt")

                if is_better_metrics(
                    va,
                    best_metrics,
                    primary_name=args.best_metric,
                    tie_name=args.best_tiebreak_metric,
                    min_delta=args.best_min_delta,
                ):
                    best_metrics = {"loss": va["loss"], "macro_f1": va["macro_f1"], "qwk": va["qwk"], "ece": va["ece"]}
                    best_epoch = epoch
                    no_improve_epochs = 0
                    if use_ema_eval and ema_state is not None:
                        ema_state.apply_to(model)
                        ckpt_best = {
                            "epoch": epoch,
                            "model_state": unwrap_model(model).state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "args": args_json,
                            "metrics": epoch_row,
                        }
                        ema_state.restore(model)
                    else:
                        ckpt_best = ckpt_last
                    torch.save(ckpt_best, args.out_dir / "stage2_best.pt")
                    if args.save_val_predictions and "ys" in va:
                        pred_rows = []
                        for y, p, probs, ev in zip(va["ys"], va["ps"], va["probs"], va["event_ids"]):
                            pred_rows.append(
                                {
                                    "y_true": int(y),
                                    "y_pred": int(p),
                                    "event_id": ev,
                                    "top_p": float(np.max(probs)),
                                    "margin": float(np.sort(np.asarray(probs))[-1] - np.sort(np.asarray(probs))[-2]),
                                    "entropy": float(-np.sum(np.asarray(probs) * np.log(np.clip(np.asarray(probs), 1e-12, 1.0)))),
                                    "probs": [float(x) for x in probs],
                                }
                            )
                        save_json(args.out_dir / "val_predictions_best.json", pred_rows)
                    if args.event_metrics and "ys" in va:
                        pe = per_event_metrics(va["ys"], va["ps"], va["event_ids"], args.classes)
                        save_json(args.out_dir / "val_event_metrics_best.json", pe)
                    print(
                        "Saved new best checkpoint.",
                        f"primary={args.best_metric}",
                        f"tie={args.best_tiebreak_metric}",
                    )
                else:
                    no_improve_epochs += 1

                save_json(args.out_dir / "metrics_history.json", history)

                if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
                    print(
                        f"Early stopping triggered at epoch {epoch}: "
                        f"no {args.best_metric} improvement for {no_improve_epochs} epochs "
                        f"(patience={args.early_stop_patience})."
                    )
                    should_stop = True
                else:
                    should_stop = False

            else:
                should_stop = False

            # Synchronize early-stop decision across all ranks to avoid DDP hang.
            if ddp:
                stop_t = torch.tensor([1 if should_stop else 0], dtype=torch.int64, device=device)
                dist.broadcast(stop_t, src=0)
                should_stop = bool(stop_t.item())

            if should_stop:
                break

        if is_main():
            elapsed = time.time() - t_train
            print(f"\nTraining complete. Elapsed seconds: {elapsed:.1f}")
            if best_metrics is not None:
                print(
                    "Best checkpoint:",
                    f"epoch={best_epoch}",
                    f"macro_f1={best_metrics['macro_f1']:.4f}",
                    f"qwk={best_metrics['qwk']:.4f}",
                    f"ece={best_metrics['ece']:.4f}",
                    f"loss={best_metrics['loss']:.5f}",
                )
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
