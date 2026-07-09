#!/usr/bin/env python3
"""Ablate the selected drop-both Stage-2a population model.

This script reuses the M23 drop-both OOF type predictions and evaluates small,
CPU-only tabular variants around the selected population model:

  core mask geometry + residual HGB on ridge geometry + 3-class soft type.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

try:
    from fit_stage2a_population_type_features import (
        _active_class_names,
        _ensure_prediction_coverage,
        _fit_extra_trees,
        _fit_hgb,
        _fit_ridge_cv,
        _fit_soft_type_experts,
        _geometry_features,
        _hard_type_features,
        _load_type_predictions,
        _metrics_for_rows,
        _parse_alphas,
        _prediction_paths,
        _prediction_rows,
        _prob_matrix,
        _target_log,
        _true_type_features,
        _type_probability_features,
    )
    from scripts.stage2a.common import (
        PROB_COLUMNS,
        parse_list,
        read_csv_rows,
        safe_float,
        save_json,
        write_csv_rows,
    )
except ImportError:  # pragma: no cover
    from II_package.scripts.stage2a.training.fit_stage2a_population_type_features import (
        _active_class_names,
        _ensure_prediction_coverage,
        _fit_extra_trees,
        _fit_hgb,
        _fit_ridge_cv,
        _fit_soft_type_experts,
        _geometry_features,
        _hard_type_features,
        _load_type_predictions,
        _metrics_for_rows,
        _parse_alphas,
        _prediction_paths,
        _prediction_rows,
        _prob_matrix,
        _target_log,
        _true_type_features,
        _type_probability_features,
    )
    from II_package.scripts.stage2a.common import (
        PROB_COLUMNS,
        parse_list,
        read_csv_rows,
        safe_float,
        save_json,
        write_csv_rows,
    )


CORE_MASK_GEOMETRY = [
    "log_footprint_m2",
    "mask_fill_ratio",
    "bbox_aspect_ratio",
    "geometry_compactness",
]
FULL_MASK_GEOMETRY = [
    "log_footprint_m2",
    "mask_area_px",
    "mask_fill_ratio",
    "bbox_w_px",
    "bbox_h_px",
    "bbox_aspect_ratio",
    "geometry_compactness",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Stage2a population ablations for a label-policy manifest.")
    p.add_argument("--labels_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--split_col", type=str, default="m20_split")
    p.add_argument("--train_split_value", type=str, default="train")
    p.add_argument("--val_split_value", type=str, default="val")
    p.add_argument("--id_col", type=str, default="building_uid")
    p.add_argument("--source_col", type=str, default="classification_source")
    p.add_argument("--type_train_predictions", type=Path, nargs="*", default=[])
    p.add_argument("--type_train_predictions_glob", type=str, default="")
    p.add_argument("--type_val_predictions", type=Path, required=True)
    p.add_argument("--type_feature_policy", type=str, default="drop_institutional_other")
    p.add_argument("--selected_geometry_cols", type=str, default=",".join(CORE_MASK_GEOMETRY))
    p.add_argument("--run_name", type=str, default="m24_drop_both_population_ablation")
    p.add_argument("--summary_basename", type=str, default="m24_drop_both_population_ablation_summary")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--ridge_alphas", type=str, default="0.01,0.1,1.0,10.0,100.0")
    p.add_argument("--hgb_max_iter", type=int, default=350)
    p.add_argument("--extra_trees_estimators", type=int, default=400)
    p.add_argument("--min_expert_effective_weight", type=float, default=80.0)
    p.add_argument("--allow_missing_type_predictions", action="store_true")
    p.add_argument("--write_prediction_csvs", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _feature_prior(pred_by_id: Mapping[str, Mapping[str, object]]) -> np.ndarray:
    prior = np.ones(len(PROB_COLUMNS), dtype=np.float64) / len(PROB_COLUMNS)
    raw_probs = []
    for pred in pred_by_id.values():
        p = np.asarray([safe_float(pred.get(col), 0.0) for col in PROB_COLUMNS], dtype=np.float64)
        if p.sum() > 0:
            raw_probs.append(p / p.sum())
    if raw_probs:
        prior = np.mean(np.vstack(raw_probs), axis=0)
        prior = prior / prior.sum()
    return prior


def _combine_features(*parts: np.ndarray) -> np.ndarray:
    usable = [p for p in parts if p.ndim == 2 and p.shape[1] > 0]
    if not usable:
        n = parts[0].shape[0] if parts else 0
        return np.zeros((n, 0), dtype=np.float64)
    return np.column_stack(usable).astype(np.float64)


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _summarize_rows(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    classes: Dict[str, int] = {}
    for row in rows:
        name = str(row.get("type_class_name") or row.get("true_type_class") or "").strip()
        if name:
            classes[name] = classes.get(name, 0) + 1
    return {"rows": len(rows), "type_class_counts": classes}


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = [
        "rank",
        "ablation_group",
        "variant",
        "model",
        "feature_set",
        "geometry_variant",
        "type_feature_set",
        "deployable",
        "selected_model",
        "population_log_mae",
        "delta_log_mae_vs_selected",
        "delta_log_mae_vs_core_hgb_geometry",
        "delta_log_mae_vs_matching_hgb_geometry",
        "population_factor2_hit_rate",
        "population_mae",
        "population_rmse",
        "population_r2",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _prediction_fieldnames(rows: Sequence[Mapping[str, object]]) -> List[str]:
    preferred = [
        "building_uid",
        "pred_population",
        "pred_log1p_population",
        "true_population",
        "true_log1p_population",
        "true_type_idx",
        "true_type_class",
        "pred_type_idx",
        "pred_type_class",
        "pred_type_conf",
        "tile_base",
        "GEOID",
        "classification_source",
    ]
    keys = set()
    for row in rows:
        keys.update(row.keys())
    ordered = [k for k in preferred if k in keys]
    ordered.extend(sorted(k for k in keys if k not in set(ordered)))
    return ordered


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = read_csv_rows(args.labels_csv)
    train_rows = [
        row
        for row in labels
        if str(row.get(args.split_col, "") or "").strip().lower() == args.train_split_value
    ]
    val_rows = [
        row
        for row in labels
        if str(row.get(args.split_col, "") or "").strip().lower() == args.val_split_value
    ]
    if not train_rows or not val_rows:
        raise RuntimeError(f"Invalid split counts for {args.split_col}: train={len(train_rows)} val={len(val_rows)}")

    train_pred_paths = _prediction_paths(args.type_train_predictions, args.type_train_predictions_glob)
    train_pred_by_id = _load_type_predictions(train_pred_paths, args.id_col)
    val_pred_by_id = _load_type_predictions([args.type_val_predictions], args.id_col)
    _ensure_prediction_coverage(train_rows, train_pred_by_id, args.id_col, "train", args.allow_missing_type_predictions)
    _ensure_prediction_coverage(val_rows, val_pred_by_id, args.id_col, "val", args.allow_missing_type_predictions)

    type_class_names = _active_class_names(args.type_feature_policy)
    train_prior = _feature_prior(train_pred_by_id)
    train_probs = _prob_matrix(train_rows, train_pred_by_id, args.id_col, train_prior, args.type_feature_policy)
    val_probs = _prob_matrix(val_rows, val_pred_by_id, args.id_col, train_prior, args.type_feature_policy)

    train_type_soft, soft_names = _type_probability_features(train_probs, type_class_names)
    val_type_soft, _ = _type_probability_features(val_probs, type_class_names)
    train_type_hard, hard_names = _hard_type_features(train_probs, type_class_names)
    val_type_hard, _ = _hard_type_features(val_probs, type_class_names)
    train_true_type, oracle_names = _true_type_features(train_rows, type_class_names)
    val_true_type, _ = _true_type_features(val_rows, type_class_names)

    selected_geometry_cols = parse_list(args.selected_geometry_cols, CORE_MASK_GEOMETRY)
    geometry_specs: Dict[str, List[str]] = {
        "core_all": selected_geometry_cols,
        "full_mask_geometry": FULL_MASK_GEOMETRY,
        "shape_only": [c for c in selected_geometry_cols if c != "log_footprint_m2"],
        "log_footprint_only": ["log_footprint_m2"],
    }
    for col in selected_geometry_cols:
        remaining = [c for c in selected_geometry_cols if c != col]
        if remaining:
            geometry_specs[f"core_drop_{col}"] = remaining
    blocked_cols = {"GEOID", "GEOID_missing", "classification_source", "estimated_units", "estimated_units_missing"}
    bad_cols = sorted(
        {
            col
            for cols in geometry_specs.values()
            for col in cols
            if col in blocked_cols or "geoid" in col.lower() or "source" in col.lower()
        }
    )
    if bad_cols:
        raise ValueError("M24 deployable ablation forbids blocked geometry/source columns: " + ",".join(bad_cols))

    type_specs: Dict[str, Tuple[np.ndarray, np.ndarray, List[str], bool, str]] = {
        "none": (
            np.zeros((len(train_rows), 0), dtype=np.float64),
            np.zeros((len(val_rows), 0), dtype=np.float64),
            [],
            True,
            "No predicted type features.",
        ),
        "soft_all": (train_type_soft, val_type_soft, list(soft_names), True, "Selected soft probability features."),
        "soft_probabilities_only": (
            train_probs,
            val_probs,
            [f"prob_{name}" for name in type_class_names],
            True,
            "Only 3-class normalized type probabilities.",
        ),
        "soft_uncertainty_only": (
            train_type_soft[:, [soft_names.index(x) for x in ("type_confidence", "type_margin", "type_entropy")]],
            val_type_soft[:, [soft_names.index(x) for x in ("type_confidence", "type_margin", "type_entropy")]],
            ["type_confidence", "type_margin", "type_entropy"],
            True,
            "Only confidence, margin, and entropy.",
        ),
        "residential_any_only": (
            train_type_soft[:, [soft_names.index("prob_residential_any")]],
            val_type_soft[:, [soft_names.index("prob_residential_any")]],
            ["prob_residential_any"],
            True,
            "Only residential-vs-commercial probability mass.",
        ),
        "hard_one_hot": (train_type_hard, val_type_hard, list(hard_names), True, "Predicted argmax type as one-hot."),
        "oracle_true_type": (train_true_type, val_true_type, list(oracle_names), False, "Ground-truth type upper bound."),
    }

    alphas = _parse_alphas(args.ridge_alphas)
    y_train = _target_log(train_rows)
    pred_dir = args.out_dir / "predictions"
    entries: List[MutableMapping[str, object]] = []
    details: Dict[str, object] = {}
    hgb_geometry_baselines: Dict[str, float] = {}

    def evaluate(
        *,
        ablation_group: str,
        variant: str,
        model_name: str,
        feature_set: str,
        geometry_variant: str,
        geometry_cols: Sequence[str],
        type_feature_set: str,
        type_feature_cols: Sequence[str],
        deployable: bool,
        selected_model: bool,
        val_pred_log: np.ndarray,
        model_meta: Mapping[str, object],
        notes: str,
    ) -> None:
        rows = _prediction_rows(val_rows, val_pred_log, val_probs, type_class_names, args.source_col, args.id_col)
        metrics = _metrics_for_rows(rows, include_source_group=False)
        run_key = f"{ablation_group}/{variant}/{model_name}"
        details[run_key] = {
            "metrics": metrics,
            "model_meta": dict(model_meta),
            "geometry_cols": list(geometry_cols),
            "type_feature_cols": list(type_feature_cols),
        }
        entry: MutableMapping[str, object] = {
            "ablation_group": ablation_group,
            "variant": variant,
            "model": model_name,
            "feature_set": feature_set,
            "geometry_variant": geometry_variant,
            "geometry_cols": list(geometry_cols),
            "type_feature_set": type_feature_set,
            "type_feature_cols": list(type_feature_cols),
            "deployable": bool(deployable),
            "selected_model": bool(selected_model),
            "population_log_mae": float(metrics["population_log_mae"]),
            "population_factor2_hit_rate": float(metrics["population_factor2_hit_rate"]),
            "population_mae": float(metrics["population_mae"]),
            "population_rmse": float(metrics["population_rmse"]),
            "population_r2": float(metrics["population_r2"]),
            "notes": notes,
        }
        entries.append(entry)
        if args.write_prediction_csvs:
            out_name = _safe_name(f"{ablation_group}_{variant}_{model_name}") + ".csv"
            write_csv_rows(pred_dir / out_name, rows, _prediction_fieldnames(rows))

    geometry_features: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = {}
    for variant, cols in geometry_specs.items():
        train_geom, geom_names = _geometry_features(train_rows, cols)
        val_geom, _ = _geometry_features(val_rows, cols)
        geometry_features[variant] = (train_geom, val_geom, geom_names)

        pred, meta = _fit_hgb(train_geom, y_train, val_geom, args.seed, args.hgb_max_iter)
        hgb_geometry_baselines[variant] = float(
            _metrics_for_rows(
                _prediction_rows(val_rows, pred, val_probs, type_class_names, args.source_col, args.id_col),
                include_source_group=False,
            )["population_log_mae"]
        )
        evaluate(
            ablation_group="geometry_baseline",
            variant=variant,
            model_name="hgb_geometry",
            feature_set="geometry",
            geometry_variant=variant,
            geometry_cols=cols,
            type_feature_set="none",
            type_feature_cols=[],
            deployable=True,
            selected_model=False,
            val_pred_log=pred,
            model_meta=meta,
            notes="Matching no-type HGB geometry baseline.",
        )

    selected_geom_train, selected_geom_val, selected_geom_names = geometry_features["core_all"]
    selected_type_train, selected_type_val, selected_type_names, selected_type_deployable, _ = type_specs["soft_all"]

    def fit_residual_hgb(
        geom_train: np.ndarray,
        geom_val: np.ndarray,
        type_train: np.ndarray,
        type_val: np.ndarray,
        geometry_cols: Sequence[str],
        type_cols: Sequence[str],
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        base_train, base_meta_train = _fit_ridge_cv(geom_train, y_train, geom_train, alphas)
        base_val, base_meta_val = _fit_ridge_cv(geom_train, y_train, geom_val, alphas)
        x_train = _combine_features(geom_train, type_train)
        x_val = _combine_features(geom_val, type_val)
        residual_y = y_train - base_train
        residual_pred, residual_meta = _fit_hgb(x_train, residual_y, x_val, args.seed + 17, args.hgb_max_iter)
        meta = {
            "base_model": "ridge_geometry",
            "base_train_meta": base_meta_train,
            "base_val_meta": base_meta_val,
            "residual_model": "hist_gradient_boosting",
            "residual_meta": residual_meta,
            "n_geometry_features": len(geometry_cols),
            "n_type_features": len(type_cols),
        }
        return base_val + residual_pred, meta

    selected_pred, selected_meta = fit_residual_hgb(
        selected_geom_train,
        selected_geom_val,
        selected_type_train,
        selected_type_val,
        selected_geometry_cols,
        selected_type_names,
    )
    evaluate(
        ablation_group="selected",
        variant="core_soft_all_residual_hgb",
        model_name="residual_hgb_on_ridge_geometry_pred_type_soft",
        feature_set="geometry_pred_type_soft_residual",
        geometry_variant="core_all",
        geometry_cols=selected_geometry_cols,
        type_feature_set="soft_all",
        type_feature_cols=selected_type_names,
        deployable=selected_type_deployable,
        selected_model=True,
        val_pred_log=selected_pred,
        model_meta=selected_meta,
        notes="M23 selected model reproduced for M24 ablation deltas.",
    )

    for variant, (geom_train, geom_val, geom_names) in geometry_features.items():
        if variant == "core_all":
            continue
        pred, meta = fit_residual_hgb(
            geom_train,
            geom_val,
            selected_type_train,
            selected_type_val,
            geometry_specs[variant],
            selected_type_names,
        )
        evaluate(
            ablation_group="geometry_ablation",
            variant=variant,
            model_name="residual_hgb_on_ridge_geometry_pred_type_soft",
            feature_set="geometry_pred_type_soft_residual",
            geometry_variant=variant,
            geometry_cols=geometry_specs[variant],
            type_feature_set="soft_all",
            type_feature_cols=selected_type_names,
            deployable=True,
            selected_model=False,
            val_pred_log=pred,
            model_meta=meta,
            notes="Change geometry columns while keeping selected residual HGB and soft type.",
        )

    zero_type_train = np.zeros((len(train_rows), 0), dtype=np.float64)
    zero_type_val = np.zeros((len(val_rows), 0), dtype=np.float64)
    for variant, (geom_train, geom_val, geom_names) in geometry_features.items():
        pred, meta = fit_residual_hgb(
            geom_train,
            geom_val,
            zero_type_train,
            zero_type_val,
            geometry_specs[variant],
            [],
        )
        evaluate(
            ablation_group="geometry_type_contribution",
            variant=f"{variant}_no_type_residual",
            model_name="residual_hgb_on_ridge_geometry_no_type",
            feature_set="geometry_no_type_residual",
            geometry_variant=variant,
            geometry_cols=geometry_specs[variant],
            type_feature_set="none",
            type_feature_cols=[],
            deployable=True,
            selected_model=False,
            val_pred_log=pred,
            model_meta=meta,
            notes="Matching residual-HGB no-type baseline for the same geometry variant.",
        )

    for type_variant, (type_train, type_val, type_names, deployable, notes) in type_specs.items():
        if type_variant == "soft_all":
            continue
        pred, meta = fit_residual_hgb(
            selected_geom_train,
            selected_geom_val,
            type_train,
            type_val,
            selected_geometry_cols,
            type_names,
        )
        evaluate(
            ablation_group="type_feature_ablation",
            variant=type_variant,
            model_name="residual_hgb_on_ridge_geometry_type_variant",
            feature_set="geometry_type_variant_residual",
            geometry_variant="core_all",
            geometry_cols=selected_geometry_cols,
            type_feature_set=type_variant,
            type_feature_cols=type_names,
            deployable=deployable,
            selected_model=False,
            val_pred_log=pred,
            model_meta=meta,
            notes=notes,
        )

    core_soft_train = _combine_features(selected_geom_train, selected_type_train)
    core_soft_val = _combine_features(selected_geom_val, selected_type_val)
    core_soft_names = [*selected_geom_names, *selected_type_names]

    ridge_pred, ridge_meta = _fit_ridge_cv(core_soft_train, y_train, core_soft_val, alphas)
    evaluate(
        ablation_group="model_form_ablation",
        variant="ridge_direct_core_soft_all",
        model_name="ridge_geometry_pred_type_soft",
        feature_set="geometry_pred_type_soft",
        geometry_variant="core_all",
        geometry_cols=selected_geometry_cols,
        type_feature_set="soft_all",
        type_feature_cols=selected_type_names,
        deployable=True,
        selected_model=False,
        val_pred_log=ridge_pred,
        model_meta={**ridge_meta, "n_features": len(core_soft_names)},
        notes="Linear direct model without residual HGB.",
    )

    hgb_pred, hgb_meta = _fit_hgb(core_soft_train, y_train, core_soft_val, args.seed, args.hgb_max_iter)
    evaluate(
        ablation_group="model_form_ablation",
        variant="hgb_direct_core_soft_all",
        model_name="hgb_geometry_pred_type_soft",
        feature_set="geometry_pred_type_soft",
        geometry_variant="core_all",
        geometry_cols=selected_geometry_cols,
        type_feature_set="soft_all",
        type_feature_cols=selected_type_names,
        deployable=True,
        selected_model=False,
        val_pred_log=hgb_pred,
        model_meta={**hgb_meta, "n_features": len(core_soft_names)},
        notes="Direct HGB model without ridge residualization.",
    )

    trees_pred, trees_meta = _fit_extra_trees(
        core_soft_train,
        y_train,
        core_soft_val,
        args.seed,
        args.extra_trees_estimators,
    )
    evaluate(
        ablation_group="model_form_ablation",
        variant="extra_trees_direct_core_soft_all",
        model_name="extra_trees_geometry_pred_type_soft",
        feature_set="geometry_pred_type_soft",
        geometry_variant="core_all",
        geometry_cols=selected_geometry_cols,
        type_feature_set="soft_all",
        type_feature_cols=selected_type_names,
        deployable=True,
        selected_model=False,
        val_pred_log=trees_pred,
        model_meta={**trees_meta, "n_features": len(core_soft_names)},
        notes="Direct bagged tree baseline.",
    )

    expert_pred, expert_meta = _fit_soft_type_experts(
        selected_geom_train,
        selected_geom_val,
        y_train,
        train_probs,
        val_probs,
        type_class_names,
        alphas,
        args.min_expert_effective_weight,
    )
    evaluate(
        ablation_group="model_form_ablation",
        variant="soft_type_experts_ridge",
        model_name="soft_type_experts_ridge",
        feature_set="geometry_soft_type_experts",
        geometry_variant="core_all",
        geometry_cols=selected_geometry_cols,
        type_feature_set="soft_expert_weights",
        type_feature_cols=[f"expert_weight_{name}" for name in type_class_names],
        deployable=True,
        selected_model=False,
        val_pred_log=expert_pred,
        model_meta=expert_meta,
        notes="Type-probability weighted ridge experts.",
    )

    selected_log_mae = next(float(x["population_log_mae"]) for x in entries if x["selected_model"])
    core_hgb_log_mae = hgb_geometry_baselines["core_all"]
    for entry in entries:
        log_mae = float(entry["population_log_mae"])
        geometry_variant = str(entry["geometry_variant"])
        matching = hgb_geometry_baselines.get(geometry_variant)
        entry["delta_log_mae_vs_selected"] = log_mae - selected_log_mae
        entry["delta_log_mae_vs_core_hgb_geometry"] = log_mae - core_hgb_log_mae
        entry["delta_log_mae_vs_matching_hgb_geometry"] = None if matching is None else log_mae - float(matching)

    entries.sort(key=lambda x: (not bool(x["deployable"]), float(x["population_log_mae"])))
    for i, entry in enumerate(entries, 1):
        entry["rank"] = i

    best_deployable = next((row for row in entries if row["deployable"]), None)
    selected_entry = next((row for row in entries if row["selected_model"]), None)
    policy_note = {
        "drop_institutional_other": "Rows use the drop_institutional_other manifest; institutional and other are absent.",
        "drop_other": "Rows use the drop_other manifest; other is absent and institutional is retained.",
        "drop_institutional": "Rows use the drop_institutional manifest; institutional is absent and other is retained.",
        "merge_institutional_other": "Rows use a merged manifest; institutional is mapped into other.",
        "raw_5class": "Rows use the raw five-class manifest.",
    }.get(args.type_feature_policy, f"Rows use type_feature_policy={args.type_feature_policy}.")

    summary = {
        "run_name": args.run_name,
        "labels_csv": str(args.labels_csv),
        "split_col": args.split_col,
        "train": _summarize_rows(train_rows),
        "validation": _summarize_rows(val_rows),
        "type_train_prediction_csvs": [str(p) for p in train_pred_paths],
        "type_val_predictions": str(args.type_val_predictions),
        "type_feature_policy": args.type_feature_policy,
        "type_feature_class_names": type_class_names,
        "selected_model_definition": {
            "model": "residual_hgb_on_ridge_geometry_pred_type_soft",
            "geometry_variant": "core_all",
            "geometry_cols": selected_geometry_cols,
            "type_feature_set": "soft_all",
            "type_feature_cols": selected_type_names,
        },
        "core_hgb_geometry_baseline_log_mae": core_hgb_log_mae,
        "selected_model": selected_entry,
        "best_deployable": best_deployable,
        "results": entries,
        "details": details,
        "input_prediction_glob_expanded_count": len(train_pred_paths),
        "notes": [
            policy_note,
            "Deltas are computed on the M20 validation split against dataset estimated_population ground truth.",
            "oracle_true_type is an upper bound and is marked non-deployable.",
        ],
    }
    save_json(args.out_dir / f"{args.summary_basename}.json", summary)
    _write_summary_csv(args.out_dir / f"{args.summary_basename}.csv", entries)
    print("[done] train_rows=", len(train_rows), "val_rows=", len(val_rows))
    print("[done] selected=", selected_entry)
    print("[done] best_deployable=", best_deployable)
    print("[done] wrote=", args.out_dir / f"{args.summary_basename}.json")


if __name__ == "__main__":
    main()
