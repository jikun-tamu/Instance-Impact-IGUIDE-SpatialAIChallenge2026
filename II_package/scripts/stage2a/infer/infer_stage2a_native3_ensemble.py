#!/usr/bin/env python3
"""Deployable Stage-2a native 3-class ensemble inference.

Pipeline:
1. Run multiple native three-logit Stage-2a type checkpoints.
2. Average calibrated-by-model softmax probabilities. No temperature scaling is
   applied by default; pass --temperatures only for an explicitly validated
   calibration run.
3. Apply the packaged `log_footprint_m2 + soft type + residual HGB` population
   artifact produced by `package_stage2a_native3_population.py`.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


def _bootstrap_import_paths() -> None:
    here = Path(__file__).resolve()
    package_root = here.parents[3]
    repo_root = package_root.parent
    for path in (str(package_root), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


_bootstrap_import_paths()

try:
    from scripts.stage2a.common import (
        DEFAULT_GEOMETRY_COLS,
        MASK_GEOMETRY_COLS,
        Stage2aInferenceDataset,
        build_model_from_config,
        class_names_from_config,
        collate_stage2a,
        geometry_vector,
        has_value,
        load_ckpt_config,
        load_state_dict_from_ckpt,
        metrics_from_prediction_rows,
        parse_list,
        read_csv_rows,
        safe_float,
        save_json,
        stage2a_prediction_fields,
    )
except ImportError:  # pragma: no cover
    from II_package.scripts.stage2a.common import (
        DEFAULT_GEOMETRY_COLS,
        MASK_GEOMETRY_COLS,
        Stage2aInferenceDataset,
        build_model_from_config,
        class_names_from_config,
        collate_stage2a,
        geometry_vector,
        has_value,
        load_ckpt_config,
        load_state_dict_from_ckpt,
        metrics_from_prediction_rows,
        parse_list,
        read_csv_rows,
        safe_float,
        save_json,
        stage2a_prediction_fields,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run selected Stage2a native3 ensemble inference.")
    p.add_argument("--input_csv", type=Path, required=True)
    p.add_argument("--ckpts", type=str, required=True, help="Comma-separated native 3-head checkpoint paths")
    p.add_argument("--population_model_pkl", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--out_metrics", type=Path, default=None)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--print_examples", type=int, default=10)
    p.add_argument("--log_every_steps", type=int, default=20)
    p.add_argument("--crop_col", type=str, default="crop_path")
    p.add_argument("--mask_col", type=str, default="mask_path")
    p.add_argument("--id_col", type=str, default="building_uid")
    p.add_argument("--geometry_cols", type=str, default="", help="Optional override for checkpoint geometry columns")
    p.add_argument(
        "--temperatures",
        type=str,
        default="",
        help="Optional comma-separated softmax temperatures aligned with --ckpts. Default is all 1.0.",
    )
    p.add_argument(
        "--allow_missing_geometry",
        action="store_true",
        help="Allow non-mask geometry inputs to fall back to missing indicators/zeros instead of failing fast.",
    )
    return p.parse_args()


def _parse_paths(text: str) -> List[Path]:
    paths = [Path(x.strip()) for x in str(text).split(",") if x.strip()]
    if not paths:
        raise ValueError("No checkpoint paths provided.")
    return paths


def _parse_temperatures(text: str, n: int) -> List[float]:
    if not str(text or "").strip():
        return [1.0] * n
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(vals) == 1 and n > 1:
        vals = vals * n
    if len(vals) != n:
        raise ValueError(f"--temperatures length {len(vals)} does not match checkpoint count {n}")
    if any(v <= 0 for v in vals):
        raise ValueError("--temperatures values must be positive")
    return vals


def _row_has_log_footprint(row: Mapping[str, object]) -> bool:
    return has_value(row.get("log_footprint_m2")) or has_value(row.get("footprint_m2"))


def _assert_deploy_geometry(rows: Sequence[Mapping[str, object]], geometry_cols: Sequence[str], allow_missing: bool) -> None:
    if allow_missing:
        return
    problems = []
    cols = list(geometry_cols)
    if "log_footprint_m2" in cols:
        missing = sum(1 for row in rows if not _row_has_log_footprint(row))
        if missing:
            problems.append(f"log_footprint_m2/footprint_m2 missing in {missing}/{len(rows)} rows")
    for col in cols:
        if col in MASK_GEOMETRY_COLS or col in {"log_footprint_m2", "estimated_units_missing", "GEOID_missing"}:
            continue
        missing = sum(1 for row in rows if not has_value(row.get(col)))
        if missing:
            problems.append(f"{col} missing in {missing}/{len(rows)} rows")
    if problems:
        raise ValueError(
            "Stage2a native3 ensemble requires deploy-visible geometry columns: "
            + "; ".join(problems)
            + ". Rebuild the inference CSV with geometry passthrough or pass --allow_missing_geometry."
        )


def _type_soft_features(probs: np.ndarray, class_names: Sequence[str]) -> tuple[np.ndarray, List[str]]:
    eps = 1e-12
    sorted_p = np.sort(probs, axis=1)
    top1 = sorted_p[:, -1]
    top2 = sorted_p[:, -2] if probs.shape[1] > 1 else np.zeros_like(top1)
    entropy = -np.sum(np.clip(probs, eps, 1.0) * np.log(np.clip(probs, eps, 1.0)), axis=1)
    cols = [probs, top1[:, None], (top1 - top2)[:, None], entropy[:, None]]
    names = [*[f"prob_{name}" for name in class_names], "type_confidence", "type_margin", "type_entropy"]
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    residential = [name_to_idx[name] for name in ("residential_small", "residential_multi") if name in name_to_idx]
    if residential:
        cols.append(np.sum(probs[:, residential], axis=1)[:, None])
        names.append("prob_residential_any")
    return np.column_stack(cols), names


class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core") :]
        if module == "numpy.random._pickle" and name == "__bit_generator_ctor":
            return _numpy_compat_bit_generator_ctor
        if module == "numpy.random._pickle" and name == "__generator_ctor":
            return _numpy_compat_generator_ctor
        return super().find_class(module, name)


class _IgnoredBitGenerator:
    def __setstate__(self, state):
        return None


def _numpy_compat_bit_generator_ctor(bit_generator_name="MT19937"):
    return _IgnoredBitGenerator()


def _numpy_compat_generator_ctor(bit_generator_name="MT19937", bit_generator_ctor=_numpy_compat_bit_generator_ctor):
    return None


def _load_population_artifact(path: Path) -> Mapping[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing population model artifact: {path}")
    with path.open("rb") as f:
        artifact = _NumpyCompatUnpickler(f).load()
    if artifact.get("format") != "stage2a_native3_population_residual_hgb_v1":
        raise ValueError(f"Unsupported population artifact format: {artifact.get('format')!r}")
    return artifact


def _predict_population(artifact: Mapping[str, object], rows: Sequence[Mapping[str, object]], probs: np.ndarray) -> np.ndarray:
    class_names = list(artifact["class_names"])
    geometry_cols = list(artifact["geometry_cols"])
    geom = np.asarray([geometry_vector(row, geometry_cols) for row in rows], dtype=np.float64)
    type_x, names = _type_soft_features(probs, class_names)
    expected = list(artifact["type_feature_names"])
    if names != expected:
        raise ValueError(f"Population type feature mismatch: expected={expected}, got={names}")
    ridge = artifact["ridge_model"]
    mean = np.asarray(artifact["ridge_mean"], dtype=np.float64).reshape(1, -1)
    std = np.asarray(artifact["ridge_std"], dtype=np.float64).reshape(1, -1)
    base = np.asarray(ridge.predict((geom - mean) / std), dtype=np.float64)
    residual_x = np.column_stack([geom, type_x])
    residual = np.asarray(artifact["residual_model"].predict(residual_x), dtype=np.float64)
    return base + residual


def _row_truth(row: Mapping[str, object], class_names: Sequence[str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    pop_raw = row.get("estimated_population")
    if not has_value(pop_raw):
        pop_raw = row.get("true_population")
    if has_value(pop_raw):
        pop = max(0.0, safe_float(pop_raw, 0.0))
        out["true_population"] = float(pop)
        out["true_log1p_population"] = float(math.log1p(pop))
    name = str(row.get("type_class_name", "") or row.get("true_type_class", "") or "").strip()
    if name in class_names:
        out["true_type_idx"] = class_names.index(name)
        out["true_type_class"] = name
    return out


def _output_rows(
    rows: Sequence[Mapping[str, object]],
    pred_log: np.ndarray,
    probs: np.ndarray,
    member_std: np.ndarray,
    class_names: Sequence[str],
    id_col: str,
    member_count: int,
) -> List[Dict[str, object]]:
    out = []
    eps = 1e-12
    entropy = -np.sum(np.clip(probs, eps, 1.0) * np.log(np.clip(probs, eps, 1.0)), axis=1)
    margin = np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]
    for i, row in enumerate(rows):
        pred_idx = int(np.argmax(probs[i]))
        item: Dict[str, object] = {
            "building_uid": row.get(id_col, row.get("building_uid", "")),
            "pred_population": float(max(0.0, math.expm1(float(pred_log[i])))),
            "pred_log1p_population": float(pred_log[i]),
            "pred_type_idx": pred_idx,
            "pred_type_class": class_names[pred_idx],
            "pred_type_conf": float(probs[i, pred_idx]),
            "crop_path": row.get("crop_path", row.get("pre_crop", "")),
            "mask_path": row.get("mask_path", row.get("mask_M", "")),
            "tile_base": row.get("tile_base", row.get("tile_id", "")),
            "GEOID": row.get("GEOID", ""),
            "classification_source": row.get("classification_source", ""),
            "ensemble_member_count": int(member_count),
            "ensemble_type_entropy": float(entropy[i]),
            "ensemble_type_margin": float(margin[i]),
        }
        for j, name in enumerate(class_names):
            item[f"prob_{name}"] = float(probs[i, j])
            item[f"prob_{name}_member_std"] = float(member_std[i, j])
        item.update(_row_truth(row, class_names))
        out.append(item)
    return out


def _fieldnames(rows: Sequence[Mapping[str, object]], class_names: Sequence[str], include_truth: bool) -> List[str]:
    fields = [
        *stage2a_prediction_fields(include_truth=include_truth, class_names=class_names),
        *[f"prob_{name}_member_std" for name in class_names],
        "ensemble_member_count",
        "ensemble_type_entropy",
        "ensemble_type_margin",
        "tile_id",
        "event_id",
        "sam3_confidence",
        "hazard_type",
    ]
    seen = set()
    ordered = []
    keys = set()
    for row in rows:
        keys.update(row.keys())
    for field in fields:
        if field in keys and field not in seen:
            ordered.append(field)
            seen.add(field)
    ordered.extend(sorted(key for key in keys if key not in seen))
    return ordered


def _run_member(
    ckpt: Path,
    config: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    args: argparse.Namespace,
    geometry_cols: Sequence[str],
    temperature: float,
    device: torch.device,
    use_cuda: bool,
) -> np.ndarray:
    ds = Stage2aInferenceDataset(
        rows,
        img_size=args.img_size,
        crop_col=args.crop_col,
        mask_col=args.mask_col,
        id_col=args.id_col,
        input_mode=str(config.get("input_mode", "rgb_mask")),
        geometry_cols=geometry_cols,
        ring_mask_col=str(config.get("ring_mask_col", "mask_R")),
        ring_radius_px=int(config.get("ring_radius_px", 48)),
        require_ring_mask=str(config.get("pooling_mode", "global")) == "mask_m_ring"
        or "ring" in str(config.get("input_mode", "rgb_mask")),
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        collate_fn=collate_stage2a,
    )
    model = build_model_from_config(config, pretrained=False).to(device)
    model.load_state_dict(load_state_dict_from_ckpt(ckpt), strict=True)
    model.eval()
    probs = []
    step = 0
    with torch.no_grad():
        for batch in loader:
            step += 1
            out = model(
                batch["x"].to(device, non_blocking=use_cuda),
                geometry=batch["geometry"].to(device, non_blocking=use_cuda),
                mask=batch["mask"].to(device, non_blocking=use_cuda),
                ring_mask=batch["ring_mask"].to(device, non_blocking=use_cuda),
                return_dict=True,
            )
            logits = out["type_logits"] / float(temperature)
            probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
            if step % max(1, args.log_every_steps) == 0:
                print(f"[infer_stage2a_native3_ensemble] {ckpt.name} step={step}/{len(loader)}")
    return np.vstack(probs).astype(np.float64)


def main() -> None:
    args = parse_args()
    rows = read_csv_rows(args.input_csv, limit=args.limit)
    if not rows:
        raise RuntimeError(f"No rows loaded from {args.input_csv}")
    for c in [args.id_col, args.crop_col, args.mask_col]:
        if c not in rows[0]:
            raise KeyError(f"Missing required column {c!r} in input CSV")
    ckpts = _parse_paths(args.ckpts)
    for ckpt in ckpts:
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing Stage2a native3 checkpoint: {ckpt}")
    temperatures = _parse_temperatures(args.temperatures, len(ckpts))
    artifact = _load_population_artifact(args.population_model_pkl)
    artifact_class_names = list(artifact["class_names"])
    configs = [load_ckpt_config(path) for path in ckpts]
    class_names = class_names_from_config(configs[0])
    if class_names != artifact_class_names:
        raise ValueError(f"Checkpoint/artifact class mismatch: ckpt={class_names}, artifact={artifact_class_names}")
    if len(class_names) != 3:
        raise ValueError(f"Expected native 3-head checkpoints, got class_names={class_names}")
    for path, cfg in zip(ckpts[1:], configs[1:]):
        names = class_names_from_config(cfg)
        if names != class_names:
            raise ValueError(f"Checkpoint {path} class_names {names} differ from first checkpoint {class_names}")
    base_config = dict(configs[0])
    base_config.setdefault("input_mode", "rgb_mask")
    base_config.setdefault("pooling_mode", "global")
    if args.geometry_cols:
        base_config["geometry_cols"] = args.geometry_cols
    base_config.setdefault("geometry_cols", DEFAULT_GEOMETRY_COLS)
    type_geometry_cols = parse_list(base_config.get("geometry_cols"), DEFAULT_GEOMETRY_COLS)
    population_geometry_cols = list(artifact["geometry_cols"])
    _assert_deploy_geometry(rows, sorted(set(type_geometry_cols) | set(population_geometry_cols)), args.allow_missing_geometry)

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("[info] rows:", len(rows))
    print("[info] device:", device)
    print("[info] class_names:", ",".join(class_names))
    print("[info] checkpoints:", len(ckpts))
    if any(abs(t - 1.0) > 1e-9 for t in temperatures):
        print("[info] applying explicit temperatures:", temperatures)
    else:
        print("[info] temperature_scaling: none")

    member_probs = []
    for i, (ckpt, cfg, temp) in enumerate(zip(ckpts, configs, temperatures), start=1):
        cfg = dict(cfg)
        cfg.setdefault("input_mode", base_config.get("input_mode", "rgb_mask"))
        cfg.setdefault("pooling_mode", base_config.get("pooling_mode", "global"))
        cfg["geometry_cols"] = ",".join(type_geometry_cols)
        print(f"[info] member {i}/{len(ckpts)} ckpt={ckpt}")
        member_probs.append(_run_member(ckpt, cfg, rows, args, type_geometry_cols, temp, device, use_cuda))
    stack = np.stack(member_probs, axis=0)
    probs = stack.mean(axis=0)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    member_std = stack.std(axis=0, ddof=0)
    pred_log = _predict_population(artifact, rows, probs)

    out_rows = _output_rows(rows, pred_log, probs, member_std, class_names, args.id_col, len(ckpts))
    for out, row in zip(out_rows, rows):
        out["tile_id"] = row.get("tile_id", row.get("tile_base", ""))
        out["event_id"] = row.get("event_id", "")
        out["sam3_confidence"] = row.get("sam3_confidence", "")
        out["hazard_type"] = row.get("hazard_type", "")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    include_truth = bool(out_rows) and all("true_population" in row for row in out_rows)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_fieldnames(out_rows, class_names, include_truth), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)
    if args.out_metrics is not None:
        if not include_truth:
            raise ValueError("--out_metrics requires rows with population/type truth.")
        save_json(args.out_metrics, metrics_from_prediction_rows(out_rows, class_names=class_names))

    print("[done] rows:", len(out_rows))
    print("[done] wrote:", args.out_csv)
    if args.out_metrics is not None:
        print("[done] wrote:", args.out_metrics)
    counts: Dict[str, int] = {}
    for row in out_rows:
        counts[str(row["pred_type_class"])] = counts.get(str(row["pred_type_class"]), 0) + 1
    pops = np.asarray([safe_float(row.get("pred_population"), 0.0) for row in out_rows], dtype=np.float64)
    print("[summary] class_counts:", counts)
    print("[summary] population_log_model:", artifact.get("model_name"))
    if len(pops):
        q = np.percentile(pops, [0, 25, 50, 75, 95, 99])
        print("[summary] pred_population quantiles p0/p25/p50/p75/p95/p99:", [round(float(x), 3) for x in q])
    for row in out_rows[: max(0, args.print_examples)]:
        print(
            "[example]",
            row["building_uid"],
            row["pred_type_class"],
            round(float(row["pred_type_conf"]), 3),
            round(float(row["pred_population"]), 3),
        )


if __name__ == "__main__":
    main()
