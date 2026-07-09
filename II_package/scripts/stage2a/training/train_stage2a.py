#!/usr/bin/env python3
"""Train Stage-2a exposure proxy models.

Supports the legacy RGB+mask multi-task baseline and the refined accuracy-track
variants: geometry fusion and soft type-conditioning. This script is designed
for reproducible experiment runs; it does not launch sweeps by itself.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from scripts.stage2a.common import (
        CLASS_NAMES,
        DEFAULT_GEOMETRY_COLS,
        BuildingPopulationModel,
        SUPPORTED_IMAGE_BACKBONES,
        SUPPORTED_STAGE2A_INPUT_MODES,
        SUPPORTED_STAGE2A_POOLING_MODES,
        Stage2aDataset,
        build_model_from_config,
        class_weights,
        collate_stage2a,
        classification_metrics,
        load_torch_checkpoint,
        metrics_from_prediction_rows,
        parse_list,
        prediction_rows_from_outputs,
        read_csv_rows,
        regression_metrics,
        save_json,
        seed_all,
        split_manifest,
        stage2a_prediction_fields,
        tile_train_val_test_split,
        write_csv_rows,
    )
except ImportError:  # pragma: no cover
    from II_package.scripts.stage2a.common import (
        CLASS_NAMES,
        DEFAULT_GEOMETRY_COLS,
        BuildingPopulationModel,
        SUPPORTED_IMAGE_BACKBONES,
        SUPPORTED_STAGE2A_INPUT_MODES,
        SUPPORTED_STAGE2A_POOLING_MODES,
        Stage2aDataset,
        build_model_from_config,
        class_weights,
        collate_stage2a,
        classification_metrics,
        load_torch_checkpoint,
        metrics_from_prediction_rows,
        parse_list,
        read_csv_rows,
        regression_metrics,
        save_json,
        seed_all,
        split_manifest,
        stage2a_prediction_fields,
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
        with cfg_path.open("r", encoding="utf-8") as f:
            defaults = json.load(f)

    p = argparse.ArgumentParser(description="Train a Stage-2a exposure proxy model.")
    p.add_argument("--config", type=Path, default=cfg_path)
    p.add_argument("--labels_csv", type=Path, required=not bool(defaults), default=defaults.get("labels_csv"))
    p.add_argument("--out_dir", type=Path, required=not bool(defaults), default=defaults.get("out_dir"))
    p.add_argument("--seed", type=int, default=int(defaults.get("seed", 2025)))
    p.add_argument("--epochs", type=int, default=int(defaults.get("epochs", 50)))
    p.add_argument("--batch_size", type=int, default=int(defaults.get("batch_size", 64)))
    p.add_argument("--lr", type=float, default=float(defaults.get("lr", 1e-4)))
    p.add_argument("--weight_decay", type=float, default=float(defaults.get("weight_decay", 0.01)))
    p.add_argument("--alpha_pop", type=float, default=float(defaults.get("alpha_pop", 1.0)))
    p.add_argument("--beta_type", type=float, default=float(defaults.get("beta_type", 0.5)))
    p.add_argument("--num_classes", type=int, default=int(defaults.get("num_classes", len(CLASS_NAMES))))
    p.add_argument(
        "--class_names",
        type=str,
        default=",".join(parse_list(defaults.get("class_names") or defaults.get("active_class_names"), [])),
        help="Comma-separated type classes for the model head. Use three names with --num_classes 3 for native drop-both training.",
    )
    p.add_argument("--loss_pop", choices=["huber", "mse", "log_cosh"], default=str(defaults.get("loss_pop", "huber")))
    p.add_argument(
        "--type_loss_mode",
        choices=["ce_baseline", "logit_adjusted_ce", "balanced_softmax", "class_balanced_focal"],
        default=str(defaults.get("type_loss_mode", "ce_baseline")),
    )
    p.add_argument("--logit_adjust_tau", type=float, default=float(defaults.get("logit_adjust_tau", 1.0)))
    p.add_argument("--focal_gamma", type=float, default=float(defaults.get("focal_gamma", 2.0)))
    p.add_argument("--class_balanced_beta", type=float, default=float(defaults.get("class_balanced_beta", 0.999)))
    p.add_argument("--class_weight_mode", choices=["none", "inverse_sqrt"], default=str(defaults.get("class_weight_mode", "inverse_sqrt")))
    p.add_argument("--val_ratio", type=float, default=float(defaults.get("val_ratio", 0.15)))
    p.add_argument("--test_ratio", type=float, default=float(defaults.get("test_ratio", 0.15)))
    p.add_argument("--amp_dtype", choices=["bf16", "fp16", "off"], default=str(defaults.get("amp_dtype", "bf16")))
    p.add_argument("--ema_decay", type=float, default=float(defaults.get("ema_decay", 0.999)))
    p.add_argument("--early_stop_patience", type=int, default=int(defaults.get("early_stop_patience", 8)))
    p.add_argument("--save_val_predictions", action="store_true", default=bool(defaults.get("save_val_predictions", True)))
    p.add_argument(
        "--backbone_name",
        choices=SUPPORTED_IMAGE_BACKBONES,
        default=str(defaults.get("backbone_name", defaults.get("backbone", "efficientnet_b0"))),
        help="Image backbone for RGB/mask Stage-2a models",
    )
    p.add_argument("--input_mode", choices=SUPPORTED_STAGE2A_INPUT_MODES, default=str(defaults.get("input_mode", "rgb_mask")))
    p.add_argument(
        "--pooling_mode",
        choices=SUPPORTED_STAGE2A_POOLING_MODES,
        default=str(defaults.get("pooling_mode", "global")),
        help="Feature pooling for image backbones: global, building mask only, or building mask plus context ring.",
    )
    p.add_argument("--task_mode", choices=["multitask", "pop_only", "type_only"], default=str(defaults.get("task_mode", "multitask")))
    p.add_argument(
        "--selection_metric",
        choices=["auto", "combined", "population_log_mae", "type_macro_f1", "loss"],
        default=str(defaults.get("selection_metric", "auto")),
        help="Validation metric used for checkpoint selection. Lower score is better; auto uses type macro-F1 for type_only.",
    )
    p.add_argument(
        "--type_conditioning_mode",
        choices=["none", "soft_probs", "detached_soft_probs", "logits", "hard_onehot"],
        default=str(defaults.get("type_conditioning_mode", "none")),
    )
    p.add_argument(
        "--type_geometry_mode",
        choices=["none", "concat"],
        default=str(defaults.get("type_geometry_mode", "none")),
        help="Whether polygon/mask geometry is concatenated into the type classifier head.",
    )
    p.add_argument("--geometry_cols", type=str, default=",".join(parse_list(defaults.get("geometry_cols"), DEFAULT_GEOMETRY_COLS)))
    p.add_argument("--split_group_col", type=str, default=str(defaults.get("split_group_col", "tile_base")))
    p.add_argument(
        "--split_col",
        type=str,
        default=str(defaults.get("split_col", "")),
        help="Optional manifest column with fixed train/val/test assignments. Preserves a precomputed blocked split after filtering rows.",
    )
    p.add_argument("--split_seed", type=int, default=int(defaults.get("split_seed", defaults.get("seed", 2025))))
    p.add_argument("--sampler_mode", choices=["natural", "weighted"], default=str(defaults.get("sampler_mode", "weighted")))
    p.add_argument("--population_clip_quantile", type=float, default=float(defaults.get("population_clip_quantile", 0.995)))
    p.add_argument("--population_bin_balance", action="store_true", default=bool(defaults.get("population_bin_balance", False)))
    p.add_argument("--img_size", type=int, default=int(defaults.get("img_size", 224)))
    p.add_argument("--ring_mask_col", type=str, default=str(defaults.get("ring_mask_col", "mask_R")))
    p.add_argument("--ring_radius_px", type=int, default=int(defaults.get("ring_radius_px", 48)))
    p.add_argument("--num_workers", type=int, default=int(defaults.get("num_workers", 4)))
    p.add_argument("--limit", type=int, default=int(defaults.get("limit", 0)))
    p.add_argument("--device", type=str, default=str(defaults.get("device", "cuda")))
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=bool(defaults.get("pretrained", True)))
    p.add_argument("--aug_hflip", type=float, default=float(defaults.get("aug_hflip", 0.5)))
    p.add_argument("--aug_vflip", type=float, default=float(defaults.get("aug_vflip", 0.0)))
    p.add_argument("--aug_rot90", type=float, default=float(defaults.get("aug_rot90", 0.25)))
    p.add_argument("--log_every_steps", type=int, default=int(defaults.get("log_every_steps", 20)))
    return p.parse_args()


def resolve_training_class_names(args: argparse.Namespace) -> List[str]:
    names = parse_list(args.class_names, [])
    if names:
        unknown = [name for name in names if name not in CLASS_NAMES]
        if unknown:
            raise ValueError(f"--class_names contains unknown Stage2a classes: {unknown}")
        return names
    if args.num_classes <= 0 or args.num_classes > len(CLASS_NAMES):
        raise ValueError(f"--num_classes must be between 1 and {len(CLASS_NAMES)}, got {args.num_classes}")
    return list(CLASS_NAMES[: int(args.num_classes)])


def split_from_manifest_column(rows: Sequence[Mapping[str, object]], split_col: str, require_test: bool = True) -> Dict[str, List[int]]:
    split: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    aliases = {
        "training": "train",
        "valid": "val",
        "validation": "val",
        "dev": "val",
        "testing": "test",
    }
    for i, row in enumerate(rows):
        raw = str(row.get(split_col, "") or "").strip().lower()
        name = aliases.get(raw, raw)
        if name not in split:
            raise ValueError(f"Row {i} has invalid split value {raw!r} in column {split_col!r}")
        split[name].append(i)
    if not split["train"] or not split["val"] or (require_test and not split["test"]):
        raise RuntimeError(f"Invalid fixed split sizes from {split_col!r}: { {k: len(v) for k, v in split.items()} }")
    return split


class EMAState:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self.backup: Dict[str, torch.Tensor] = {}

    def update(self, model: nn.Module) -> None:
        if self.decay <= 0:
            return
        state = model.state_dict()
        for k, v in state.items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> None:
        self.backup = {}
        state = model.state_dict()
        for k, v in state.items():
            if k in self.shadow:
                self.backup[k] = v.detach().clone()
                v.copy_(self.shadow[k])

    def restore(self, model: nn.Module) -> None:
        state = model.state_dict()
        for k, v in self.backup.items():
            state[k].copy_(v)
        self.backup = {}


def pop_loss_fn(pred: torch.Tensor, target: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "mse":
        return (pred - target).pow(2)
    if mode == "log_cosh":
        x = pred - target
        return x + F.softplus(-2.0 * x) - math.log(2.0)
    return torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")


def type_loss_fn(
    logits: torch.Tensor,
    target: torch.Tensor,
    args: argparse.Namespace,
    class_weight_tensor: Optional[torch.Tensor] = None,
    class_count_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if args.type_loss_mode == "logit_adjusted_ce":
        if class_count_tensor is None:
            raise ValueError("class_count_tensor is required for logit_adjusted_ce")
        prior = class_count_tensor / torch.clamp(class_count_tensor.sum(), min=1.0)
        adjusted = logits + float(args.logit_adjust_tau) * torch.log(torch.clamp(prior, min=1e-12)).view(1, -1)
        return torch.nn.functional.cross_entropy(adjusted, target, reduction="mean")
    if args.type_loss_mode == "balanced_softmax":
        if class_count_tensor is None:
            raise ValueError("class_count_tensor is required for balanced_softmax")
        adjusted = logits + torch.log(torch.clamp(class_count_tensor, min=1.0)).view(1, -1)
        return torch.nn.functional.cross_entropy(adjusted, target, reduction="mean")
    if args.type_loss_mode == "class_balanced_focal":
        if class_count_tensor is None:
            raise ValueError("class_count_tensor is required for class_balanced_focal")
        beta = float(args.class_balanced_beta)
        effective_num = 1.0 - torch.pow(torch.full_like(class_count_tensor, beta), torch.clamp(class_count_tensor, min=1.0))
        weights = (1.0 - beta) / torch.clamp(effective_num, min=1e-12)
        weights = weights / torch.clamp(weights.mean(), min=1e-12)
        ce = torch.nn.functional.cross_entropy(logits, target, weight=weights, reduction="none")
        pt = torch.exp(-ce)
        focal = torch.pow(1.0 - pt, float(args.focal_gamma)) * ce
        return focal.mean()
    return torch.nn.functional.cross_entropy(
        logits,
        target,
        weight=class_weight_tensor,
        reduction="mean",
    )


def make_loader(
    rows: Sequence[Mapping[str, object]],
    indices: Sequence[int],
    args: argparse.Namespace,
    train: bool,
    sampler: Optional[WeightedRandomSampler] = None,
) -> DataLoader:
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


def make_weighted_sampler(rows: Sequence[Mapping[str, object]], indices: Sequence[int], args: argparse.Namespace) -> WeightedRandomSampler:
    weights = np.asarray(class_weights(rows, indices, alpha=0.5, cap=10.0), dtype=np.float64)
    if args.population_bin_balance:
        pops = np.asarray([float(rows[i].get("estimated_population", 0.0) or 0.0) for i in indices], dtype=np.float64)
        bins = np.digitize(np.log1p(pops), np.quantile(np.log1p(pops), [0.25, 0.5, 0.75, 0.9]))
        counts = {b: max(1, int(np.sum(bins == b))) for b in np.unique(bins)}
        max_count = max(counts.values())
        weights *= np.asarray([(max_count / counts[b]) ** 0.5 for b in bins], dtype=np.float64)
    return WeightedRandomSampler(weights=torch.as_tensor(weights, dtype=torch.double), num_samples=len(indices), replacement=True)


def _autocast_dtype(args: argparse.Namespace, device: torch.device):
    if device.type != "cuda" or args.amp_dtype == "off":
        return None
    return torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16


def run_epoch(
    model: BuildingPopulationModel,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    class_weight_tensor: Optional[torch.Tensor] = None,
    class_count_tensor: Optional[torch.Tensor] = None,
    pop_clip_value: Optional[float] = None,
    ema_state: Optional[EMAState] = None,
    collect_outputs: bool = False,
    class_names: Sequence[str] = CLASS_NAMES,
) -> Dict[str, object]:
    train = optimizer is not None
    model.train(train)
    ac_dtype = _autocast_dtype(args, device)
    all_pop_log: List[np.ndarray] = []
    all_pop_target: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_type: List[np.ndarray] = []
    all_metas: List[Mapping[str, object]] = []
    total_loss = 0.0
    n_seen = 0

    for step, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        ring_mask = batch["ring_mask"].to(device, non_blocking=True)
        geometry = batch["geometry"].to(device, non_blocking=True)
        pop_t = batch["pop_log"].to(device, non_blocking=True)
        type_t = batch["type_idx"].to(device, non_blocking=True)
        if pop_clip_value is not None:
            pop_t_loss = torch.clamp(pop_t, max=float(pop_clip_value))
        else:
            pop_t_loss = pop_t

        with torch.autocast(device_type=device.type, dtype=ac_dtype, enabled=ac_dtype is not None):
            out = model(x, geometry=geometry, mask=mask, ring_mask=ring_mask, return_dict=True)
            loss_parts = []
            if args.task_mode != "type_only":
                lp = pop_loss_fn(out["pop_log"], pop_t_loss, args.loss_pop)
                loss_parts.append(args.alpha_pop * lp.mean())
            if args.task_mode != "pop_only":
                ce = type_loss_fn(
                    out["type_logits"],
                    type_t,
                    args,
                    class_weight_tensor=class_weight_tensor,
                    class_count_tensor=class_count_tensor,
                )
                loss_parts.append(args.beta_type * ce)
            loss = sum(loss_parts) if loss_parts else out["pop_log"].sum() * 0.0

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if ema_state is not None:
                ema_state.update(model)

        bsz = int(x.shape[0])
        total_loss += float(loss.detach().cpu()) * bsz
        n_seen += bsz
        if collect_outputs:
            all_pop_log.append(out["pop_log"].detach().float().cpu().numpy())
            all_pop_target.append(pop_t.detach().float().cpu().numpy())
            all_probs.append(out["type_probs"].detach().float().cpu().numpy())
            all_type.append(type_t.detach().cpu().numpy())
            all_metas.extend(batch["meta"])
        if train and step % max(1, args.log_every_steps) == 0:
            print(f"[train_stage2a] step={step}/{len(loader)} loss={total_loss / max(1, n_seen):.5f}")

    metrics: Dict[str, object] = {"loss": total_loss / max(1, n_seen)}
    if collect_outputs and all_pop_log:
        pop_log = np.concatenate(all_pop_log)
        pop_target = np.concatenate(all_pop_target)
        probs = np.concatenate(all_probs)
        y_type = np.concatenate(all_type)
        metrics.update(regression_metrics(pop_target, pop_log))
        metrics.update(classification_metrics(y_type, probs=probs, class_names=class_names))
        metrics["prediction_rows"] = prediction_rows_from_outputs(all_metas, pop_log, probs, pop_target, y_type, class_names=class_names)
    return metrics


def resolve_selection_metric(task_mode: str, selection_metric: str = "auto") -> str:
    if selection_metric != "auto":
        return selection_metric
    if task_mode == "type_only":
        return "type_macro_f1"
    if task_mode == "pop_only":
        return "population_log_mae"
    return "combined"


def combined_selection_score(
    metrics: Mapping[str, object],
    task_mode: str = "multitask",
    selection_metric: str = "auto",
) -> float:
    """Return a validation checkpoint score where lower is always better."""
    resolved = resolve_selection_metric(task_mode, selection_metric)
    if resolved == "type_macro_f1":
        return -float(metrics.get("type_macro_f1", 0.0))
    if resolved == "population_log_mae":
        return float(metrics.get("population_log_mae", metrics.get("loss", 0.0)))
    if resolved == "loss":
        return float(metrics.get("loss", 0.0))
    # Combined: population log error is primary; type macro-F1 is a small tie helper.
    return float(metrics.get("population_log_mae", metrics.get("loss", 0.0))) - 0.05 * float(metrics.get("type_macro_f1", 0.0))


def checkpoint_config(args: argparse.Namespace) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def save_checkpoint(
    path: Path,
    model: BuildingPopulationModel,
    args: argparse.Namespace,
    metrics: Mapping[str, object],
    class_names: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": checkpoint_config(args),
            "metrics": {k: v for k, v in metrics.items() if k != "prediction_rows"},
            "class_names": list(class_names),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    class_names = resolve_training_class_names(args)
    args.num_classes = len(class_names)
    args.class_names = ",".join(class_names)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    rows = read_csv_rows(args.labels_csv, limit=args.limit)
    if args.split_col:
        split = split_from_manifest_column(rows, args.split_col, require_test=args.test_ratio > 0)
    else:
        split = tile_train_val_test_split(rows, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.split_seed, tile_col=args.split_group_col)
    save_json(args.out_dir / "train_config.json", vars(args))
    split_meta = split_manifest(rows, split, args.split_group_col)
    if args.split_col:
        split_meta["assignment_col"] = args.split_col
        split_meta["assignment_source"] = "manifest_column"
    save_json(args.out_dir / "split_manifest.json", split_meta)

    pops = np.asarray([float(rows[i].get("estimated_population", 0.0) or 0.0) for i in split["train"]], dtype=np.float64)
    pop_clip_value = None
    if 0.0 < args.population_clip_quantile < 1.0 and len(pops):
        pop_clip_value = float(np.log1p(np.quantile(pops, args.population_clip_quantile)))
    save_json(
        args.out_dir / "target_stats.json",
        {
            "population_clip_quantile": args.population_clip_quantile,
            "population_clip_log1p_value": pop_clip_value,
            "population_clip_source": "train_split_only",
            "train_population_count": int(len(pops)),
        },
    )

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("[info] rows=", len(rows), "split=", {k: len(v) for k, v in split.items()}, "device=", device)

    sampler = make_weighted_sampler(rows, split["train"], args) if args.sampler_mode == "weighted" else None
    train_loader = make_loader(rows, split["train"], args, train=True, sampler=sampler)
    val_loader = make_loader(rows, split["val"], args, train=False)
    test_loader = make_loader(rows, split["test"], args, train=False) if split.get("test") else None

    model = build_model_from_config(vars(args), pretrained=args.pretrained).to(device)

    class_count_tensor = None
    train_labels = [int(rows[i].get("type_class", 0)) for i in split["train"]]
    invalid_labels = sorted({y for y in train_labels if y < 0 or y >= args.num_classes})
    if invalid_labels:
        raise RuntimeError(
            f"Training labels contain indices outside native head range 0..{args.num_classes - 1}: {invalid_labels}. "
            "Use a remapped manifest or a compatibility head for non-prefix class policies."
        )
    train_counts = np.asarray([sum(1 for y in train_labels if y == c) for c in range(args.num_classes)], dtype=np.float32)
    class_count_tensor = torch.as_tensor(train_counts, dtype=torch.float32, device=device)
    save_json(args.out_dir / "type_class_counts.json", {class_names[i]: int(train_counts[i]) for i in range(len(class_names))})

    weights = None
    if args.class_weight_mode == "inverse_sqrt":
        train_w = class_weights(rows, split["train"], alpha=0.5, cap=10.0)
        per_class = np.ones(args.num_classes, dtype=np.float32)
        for c in range(args.num_classes):
            vals = [w for w, y in zip(train_w, train_labels) if y == c]
            if vals:
                per_class[c] = float(np.mean(vals))
        weights = torch.as_tensor(per_class, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp_dtype == "fp16" and device.type == "cuda"))
    ema_state = EMAState(model, args.ema_decay) if args.ema_decay > 0 else None
    best_score = float("inf")
    best_epoch = 0
    no_improve = 0
    history_path = args.out_dir / "metrics_history.jsonl"
    if history_path.exists():
        history_path.unlink()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr = run_epoch(model, train_loader, args, device, optimizer, scaler, weights, class_count_tensor, pop_clip_value, ema_state, False, class_names)
        ema_applied = ema_state is not None and epoch > 1
        if ema_applied:
            ema_state.apply_to(model)
        va = run_epoch(model, val_loader, args, device, class_count_tensor=class_count_tensor, collect_outputs=True, class_names=class_names)
        resolved_selection = resolve_selection_metric(args.task_mode, args.selection_metric)
        score = combined_selection_score(va, task_mode=args.task_mode, selection_metric=args.selection_metric)
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            **{k: v for k, v in va.items() if k != "prediction_rows"},
            "selection_metric": resolved_selection,
            "selection_score": score,
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            "epoch",
            epoch,
            "| train_loss",
            f"{tr['loss']:.5f}",
            "| val_log_mae",
            f"{float(va.get('population_log_mae', 0.0)):.4f}",
            "| val_type_macro_f1",
            f"{float(va.get('type_macro_f1', 0.0)):.4f}",
            "| selection",
            resolved_selection,
            "| score",
            f"{score:.4f}",
        )
        save_checkpoint(args.out_dir / "stage2a_last_model.pt", model, args, va, class_names)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(args.out_dir / "stage2a_best_model.pt", model, args, va, class_names)
            if args.save_val_predictions:
                write_csv_rows(args.out_dir / "best_val_predictions.csv", va["prediction_rows"], stage2a_prediction_fields(True, class_names=class_names))
        else:
            no_improve += 1
        if ema_applied:
            ema_state.restore(model)
        if no_improve >= args.early_stop_patience:
            print("[early_stop] no improvement for", no_improve, "epochs")
            break

    state = load_torch_checkpoint(args.out_dir / "stage2a_best_model.pt", map_location=device)
    model.load_state_dict(state["model_state"])
    test_metrics = None
    if test_loader is not None:
        te = run_epoch(model, test_loader, args, device, class_count_tensor=class_count_tensor, collect_outputs=True, class_names=class_names)
        test_metrics = {k: v for k, v in te.items() if k != "prediction_rows"}
        save_json(args.out_dir / "test_metrics.json", test_metrics)
        write_csv_rows(args.out_dir / "test_predictions.csv", te["prediction_rows"], stage2a_prediction_fields(True, class_names=class_names))
    save_json(
        args.out_dir / "summary.json",
        {
            "best_epoch": best_epoch,
            "best_score": best_score,
            "selection_metric": resolve_selection_metric(args.task_mode, args.selection_metric),
            "validation_metrics": state.get("metrics", {}),
            "test_metrics": test_metrics,
            "has_test_split": test_loader is not None,
        },
    )
    print("[done] wrote train_config.json")
    print("[done] wrote split_manifest.json")
    print("[done] wrote stage2a_last_model.pt")
    print("[done] wrote stage2a_best_model.pt")
    if test_loader is not None:
        print("[done] wrote test_metrics.json")
    else:
        print("[done] skipped test_metrics.json because test split is empty")


if __name__ == "__main__":
    main()
