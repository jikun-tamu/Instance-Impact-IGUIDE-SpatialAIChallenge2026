#!/usr/bin/env python3
"""Fit Stage-2a population regressors that use predicted type probabilities."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

try:
    from scripts.stage2a.common import (
        CLASS_NAMES,
        DEFAULT_GEOMETRY_COLS,
        PROB_COLUMNS,
        geometry_vector,
        grouped_regression_metrics,
        parse_list,
        read_csv_rows,
        regression_metrics,
        safe_float,
        safe_int,
        save_json,
        write_csv_rows,
    )
except ImportError:  # pragma: no cover
    from II_package.scripts.stage2a.common import (
        CLASS_NAMES,
        DEFAULT_GEOMETRY_COLS,
        PROB_COLUMNS,
        geometry_vector,
        grouped_regression_metrics,
        parse_list,
        read_csv_rows,
        regression_metrics,
        safe_float,
        safe_int,
        save_json,
        write_csv_rows,
    )


TYPE_FEATURE_POLICIES = {
    "raw_5class": list(CLASS_NAMES),
    "merge_institutional_other": ["residential_small", "residential_multi", "commercial", "other"],
    "drop_institutional": ["residential_small", "residential_multi", "commercial", "other"],
    "drop_other": ["residential_small", "residential_multi", "commercial", "institutional"],
    "drop_institutional_other": ["residential_small", "residential_multi", "commercial"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate population models with predicted Stage2a type features.")
    p.add_argument("--labels_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--split_col", type=str, default="m20_split")
    p.add_argument("--train_split_value", type=str, default="train")
    p.add_argument("--val_split_value", type=str, default="val")
    p.add_argument("--id_col", type=str, default="building_uid")
    p.add_argument("--source_col", type=str, default="classification_source")
    p.add_argument("--geometry_cols", type=str, default=",".join(DEFAULT_GEOMETRY_COLS))
    p.add_argument("--type_train_predictions", type=Path, nargs="*", default=[])
    p.add_argument("--type_train_predictions_glob", type=str, default="")
    p.add_argument("--type_val_predictions", type=Path, required=True)
    p.add_argument(
        "--type_feature_policy",
        choices=sorted(TYPE_FEATURE_POLICIES),
        default="raw_5class",
        help=(
            "Projection applied to type prediction probabilities before building population features. "
            "Use drop_institutional_other for the reduced 3-class Stage2a taxonomy."
        ),
    )
    p.add_argument("--run_name", type=str, default="m21_population_type_features")
    p.add_argument(
        "--feature_contract",
        choices=["source_aware", "deployable_image"],
        default="source_aware",
        help=(
            "source_aware reproduces M21 with classification_source/source-aware features. "
            "deployable_image excludes classification_source/GEOID-style metadata and uses only image/mask-derived "
            "geometry plus predicted type features."
        ),
    )
    p.add_argument("--summary_basename", type=str, default="m21_population_type_feature_summary")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--ridge_alphas", type=str, default="0.01,0.1,1.0,10.0,100.0")
    p.add_argument("--hgb_max_iter", type=int, default=350)
    p.add_argument("--extra_trees_estimators", type=int, default=400)
    p.add_argument("--min_source_train_rows", type=int, default=100)
    p.add_argument("--min_expert_effective_weight", type=float, default=80.0)
    p.add_argument("--allow_missing_type_predictions", action="store_true")
    p.add_argument("--no_oracle", action="store_true")
    p.add_argument("--write_prediction_csvs", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _row_id(row: Mapping[str, object], id_col: str) -> str:
    return str(row.get(id_col, "") or "").strip()


def _target_log(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    return np.asarray([math.log1p(max(0.0, safe_float(row.get("estimated_population"), 0.0))) for row in rows], dtype=np.float64)


def _source_value(row: Mapping[str, object], source_col: str) -> str:
    value = str(row.get(source_col, "") or "").strip()
    return value if value else "__missing__"


def _parse_alphas(text: str) -> List[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("--ridge_alphas produced no values")
    return vals


def _prediction_paths(paths: Sequence[Path], pattern: str) -> List[Path]:
    out = [Path(p) for p in paths]
    if pattern:
        out.extend(Path(p) for p in sorted(glob.glob(pattern)))
    unique: List[Path] = []
    seen = set()
    for p in out:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    if not unique:
        raise RuntimeError("No type prediction CSVs were provided.")
    return unique


def _load_type_predictions(paths: Sequence[Path], id_col: str) -> Dict[str, Dict[str, str]]:
    by_id: Dict[str, Dict[str, str]] = {}
    duplicates: List[str] = []
    for path in paths:
        for row in read_csv_rows(path):
            bid = _row_id(row, id_col)
            if not bid:
                continue
            if bid in by_id:
                duplicates.append(bid)
                continue
            out = dict(row)
            out["_prediction_source_csv"] = str(path)
            by_id[bid] = out
    if duplicates:
        sample = ", ".join(sorted(set(duplicates))[:5])
        raise RuntimeError(f"Duplicate type predictions for {len(set(duplicates))} building ids; examples: {sample}")
    return by_id


def _active_class_names(policy: str) -> List[str]:
    if policy not in TYPE_FEATURE_POLICIES:
        raise ValueError(f"Unsupported type feature policy {policy!r}")
    return list(TYPE_FEATURE_POLICIES[policy])


def _raw_probability_vector(pred: Mapping[str, object], fallback_prior: np.ndarray) -> np.ndarray:
    p = np.asarray([safe_float(pred.get(col), 0.0) for col in PROB_COLUMNS], dtype=np.float64)
    if float(p.sum()) > 0:
        return p / float(p.sum())
    idx = safe_int(pred.get("pred_type_idx"), -1)
    p = np.zeros(len(PROB_COLUMNS), dtype=np.float64)
    if 0 <= idx < len(p):
        p[idx] = 1.0
        return p
    return np.asarray(fallback_prior, dtype=np.float64).copy()


def _project_type_probs(raw_probs: np.ndarray, policy: str) -> np.ndarray:
    if policy == "raw_5class":
        out = np.asarray(raw_probs, dtype=np.float64)
    elif policy == "merge_institutional_other":
        out = np.asarray([raw_probs[0], raw_probs[1], raw_probs[2], raw_probs[3] + raw_probs[4]], dtype=np.float64)
    elif policy == "drop_institutional":
        out = np.asarray([raw_probs[0], raw_probs[1], raw_probs[2], raw_probs[4]], dtype=np.float64)
    elif policy == "drop_other":
        out = np.asarray([raw_probs[0], raw_probs[1], raw_probs[2], raw_probs[3]], dtype=np.float64)
    elif policy == "drop_institutional_other":
        out = np.asarray([raw_probs[0], raw_probs[1], raw_probs[2]], dtype=np.float64)
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unsupported type feature policy {policy!r}")
    total = float(out.sum())
    if total <= 0:
        return np.ones(len(out), dtype=np.float64) / max(1, len(out))
    return out / total


def _ensure_prediction_coverage(
    rows: Sequence[Mapping[str, object]],
    pred_by_id: Mapping[str, Mapping[str, object]],
    id_col: str,
    split_name: str,
    allow_missing: bool,
) -> None:
    missing = [_row_id(row, id_col) for row in rows if _row_id(row, id_col) not in pred_by_id]
    if missing and not allow_missing:
        sample = ", ".join(missing[:5])
        raise RuntimeError(f"Missing {len(missing)} {split_name} type predictions; examples: {sample}")


def _prob_matrix(
    rows: Sequence[Mapping[str, object]],
    pred_by_id: Mapping[str, Mapping[str, object]],
    id_col: str,
    fallback_prior: np.ndarray,
    policy: str,
) -> np.ndarray:
    probs = []
    for row in rows:
        pred = pred_by_id.get(_row_id(row, id_col))
        if pred is None:
            p = _project_type_probs(fallback_prior, policy)
        else:
            p = _project_type_probs(_raw_probability_vector(pred, fallback_prior), policy)
        probs.append(p)
    return np.vstack(probs).astype(np.float64)


def _type_probability_features(probs: np.ndarray, class_names: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    eps = 1e-12
    sorted_p = np.sort(probs, axis=1)
    top1 = sorted_p[:, -1]
    top2 = sorted_p[:, -2] if probs.shape[1] > 1 else np.zeros_like(top1)
    entropy = -np.sum(np.clip(probs, eps, 1.0) * np.log(np.clip(probs, eps, 1.0)), axis=1)
    extra_cols = [top1, top1 - top2, entropy]
    extra_names = ["type_confidence", "type_margin", "type_entropy"]
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    residential_indices = [name_to_idx[name] for name in ("residential_small", "residential_multi") if name in name_to_idx]
    if residential_indices:
        extra_cols.append(np.sum(probs[:, residential_indices], axis=1))
        extra_names.append("prob_residential_any")
    ambiguous_indices = [name_to_idx[name] for name in ("institutional", "other") if name in name_to_idx]
    if ambiguous_indices:
        extra_cols.append(np.sum(probs[:, ambiguous_indices], axis=1))
        extra_names.append("prob_institutional_plus_other")
    extra = np.column_stack(extra_cols)
    names = [*[f"prob_{name}" for name in class_names], *extra_names]
    return np.column_stack([probs, extra]), names


def _hard_type_features(probs: np.ndarray, class_names: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    idx = np.argmax(probs, axis=1)
    hard = np.zeros_like(probs)
    hard[np.arange(len(idx)), idx] = 1.0
    names = [f"pred_hard_{name}" for name in class_names]
    return hard, names


def _row_class_name(row: Mapping[str, object]) -> str:
    name = str(row.get("type_class_name", "") or row.get("true_type_class", "") or "").strip()
    if name:
        return name
    idx = safe_int(row.get("type_class", row.get("true_type_idx")), -1)
    return CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else ""


def _true_type_features(rows: Sequence[Mapping[str, object]], class_names: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    arr = np.zeros((len(rows), len(class_names)), dtype=np.float64)
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    for i, row in enumerate(rows):
        name = _row_class_name(row)
        idx = name_to_idx.get(name, -1)
        if 0 <= idx < len(class_names):
            arr[i, idx] = 1.0
    return arr, [f"oracle_true_{name}" for name in class_names]


def _geometry_features(rows: Sequence[Mapping[str, object]], cols: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    return np.asarray([geometry_vector(row, cols) for row in rows], dtype=np.float64), [f"geom_{c}" for c in cols]


def _source_onehot(
    train_rows: Sequence[Mapping[str, object]],
    val_rows: Sequence[Mapping[str, object]],
    source_col: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    categories = sorted({_source_value(row, source_col) for row in train_rows})
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    names = [f"source_{c}" for c in categories] + ["source___unknown__"]

    def encode(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
        x = np.zeros((len(rows), len(names)), dtype=np.float64)
        for i, row in enumerate(rows):
            source = _source_value(row, source_col)
            j = cat_to_idx.get(source, len(names) - 1)
            x[i, j] = 1.0
        return x

    return encode(train_rows), encode(val_rows), names, categories


def _standardize(train_x: np.ndarray, val_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0, keepdims=True)
    sd = train_x.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (train_x - mu) / sd, (val_x - mu) / sd, mu.squeeze(0), sd.squeeze(0)


def _fit_ridge_cv(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, alphas: Sequence[float], sample_weight=None):
    from sklearn.linear_model import RidgeCV

    train_xs, val_xs, mu, sd = _standardize(train_x, val_x)
    cv = min(5, max(2, len(train_y) // 1000))
    model = RidgeCV(alphas=list(alphas), cv=cv)
    if sample_weight is None:
        model.fit(train_xs, train_y)
    else:
        model.fit(train_xs, train_y, sample_weight=sample_weight)
    pred = np.asarray(model.predict(val_xs), dtype=np.float64)
    return pred, {"alpha": float(model.alpha_), "feature_mean": mu.tolist(), "feature_std": sd.tolist()}


def _fit_hgb(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, seed: int, max_iter: int) -> Tuple[np.ndarray, Dict[str, object]]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    try:
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=max_iter,
            learning_rate=0.04,
            l2_regularization=0.05,
            min_samples_leaf=25,
            random_state=seed,
        )
        loss = "absolute_error"
    except TypeError:  # pragma: no cover - older sklearn fallback
        model = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=0.04, random_state=seed)
        loss = "default"
    model.fit(train_x, train_y)
    return np.asarray(model.predict(val_x), dtype=np.float64), {"loss": loss, "max_iter": max_iter}


def _fit_extra_trees(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    seed: int,
    n_estimators: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    from sklearn.ensemble import ExtraTreesRegressor

    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        max_features=0.85,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(train_x, train_y)
    return np.asarray(model.predict(val_x), dtype=np.float64), {"n_estimators": n_estimators, "min_samples_leaf": 2}


def _fit_source_aware_ridge(
    train_x: np.ndarray,
    val_x: np.ndarray,
    train_y: np.ndarray,
    train_rows: Sequence[Mapping[str, object]],
    val_rows: Sequence[Mapping[str, object]],
    source_col: str,
    alphas: Sequence[float],
    min_source_train_rows: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    global_pred, global_meta = _fit_ridge_cv(train_x, train_y, val_x, alphas)
    pred = np.asarray(global_pred, dtype=np.float64)
    meta: Dict[str, object] = {"fallback": global_meta, "sources": {}}
    train_sources = np.asarray([_source_value(row, source_col) for row in train_rows])
    val_sources = np.asarray([_source_value(row, source_col) for row in val_rows])
    for source in sorted(set(train_sources) | set(val_sources)):
        train_mask = train_sources == source
        val_mask = val_sources == source
        if not np.any(val_mask):
            continue
        used_specific = int(np.sum(train_mask)) >= int(min_source_train_rows)
        if used_specific:
            source_pred, source_meta = _fit_ridge_cv(train_x[train_mask], train_y[train_mask], val_x[val_mask], alphas)
            pred[val_mask] = source_pred
        else:
            source_meta = dict(global_meta)
        meta["sources"][source] = {
            "train_rows": int(np.sum(train_mask)),
            "val_rows": int(np.sum(val_mask)),
            "used_source_specific_ridge": bool(used_specific),
            **source_meta,
        }
    return pred, meta


def _fit_soft_type_experts(
    train_x: np.ndarray,
    val_x: np.ndarray,
    train_y: np.ndarray,
    train_probs: np.ndarray,
    val_probs: np.ndarray,
    class_names: Sequence[str],
    alphas: Sequence[float],
    min_effective_weight: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    fallback_pred, fallback_meta = _fit_ridge_cv(train_x, train_y, val_x, alphas)
    expert_preds = []
    experts = {}
    for k, name in enumerate(class_names):
        weights = np.asarray(train_probs[:, k], dtype=np.float64)
        effective = float(weights.sum())
        if effective >= float(min_effective_weight):
            pred_k, meta_k = _fit_ridge_cv(train_x, train_y, val_x, alphas, sample_weight=weights)
            used = True
        else:
            pred_k = fallback_pred
            meta_k = dict(fallback_meta)
            used = False
        expert_preds.append(pred_k)
        experts[name] = {"effective_weight": effective, "used_specific_expert": bool(used), **meta_k}
    stacked = np.vstack(expert_preds).T
    pred = np.sum(val_probs * stacked, axis=1)
    return pred, {"fallback": fallback_meta, "experts": experts}


def _prediction_rows(
    val_rows: Sequence[Mapping[str, object]],
    val_pred_log: np.ndarray,
    val_probs: np.ndarray,
    class_names: Sequence[str],
    source_col: str,
    id_col: str,
) -> List[Dict[str, object]]:
    out = []
    pred_pop = np.maximum(0.0, np.expm1(val_pred_log))
    for i, row in enumerate(val_rows):
        true_pop = max(0.0, safe_float(row.get("estimated_population"), 0.0))
        true_name = _row_class_name(row)
        true_idx = class_names.index(true_name) if true_name in class_names else -1
        pred_type_idx = int(np.argmax(val_probs[i]))
        item: Dict[str, object] = {
            "building_uid": _row_id(row, id_col),
            "pred_population": float(pred_pop[i]),
            "pred_log1p_population": float(val_pred_log[i]),
            "true_population": float(true_pop),
            "true_log1p_population": float(math.log1p(true_pop)),
            "true_type_idx": true_idx,
            "true_type_class": class_names[true_idx] if 0 <= true_idx < len(class_names) else "",
            "pred_type_idx": pred_type_idx,
            "pred_type_class": class_names[pred_type_idx],
            "pred_type_conf": float(np.max(val_probs[i])),
            "tile_base": row.get("tile_base", ""),
            "GEOID": row.get("GEOID", ""),
            "classification_source": _source_value(row, source_col),
        }
        for j, col in enumerate(f"prob_{name}" for name in class_names):
            item[col] = float(val_probs[i, j])
        out.append(item)
    return out


def _metrics_for_rows(rows: Sequence[Mapping[str, object]], include_source_group: bool = True) -> Dict[str, object]:
    y_true = [safe_float(row.get("true_population"), 0.0) for row in rows]
    y_pred = [safe_float(row.get("pred_population"), 0.0) for row in rows]
    metrics: Dict[str, object] = regression_metrics(np.log1p(y_true), np.log1p(y_pred), y_true, y_pred)
    metrics["population_by_type"] = grouped_regression_metrics(rows, "true_type_class")
    if include_source_group:
        metrics["population_by_classification_source"] = grouped_regression_metrics(rows, "classification_source")
    bin_rows = []
    for row, pop in zip(rows, y_true):
        r2 = dict(row)
        if pop <= 0:
            bin_name = "zero"
        elif pop <= 100:
            bin_name = "lt_100"
        elif pop <= 300:
            bin_name = "100_300"
        elif pop <= 1000:
            bin_name = "300_1000"
        else:
            bin_name = "gt_1000"
        r2["population_bin"] = bin_name
        bin_rows.append(r2)
    metrics["population_by_bin"] = grouped_regression_metrics(bin_rows, "population_bin")
    return metrics


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = [
        "model",
        "feature_set",
        "deployable",
        "baseline_model",
        "population_log_mae",
        "delta_log_mae_vs_baseline",
        "delta_log_mae_vs_source_aware_geometry",
        "population_factor2_hit_rate",
        "population_mae",
        "population_rmse",
        "population_r2",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = read_csv_rows(args.labels_csv)
    train_rows = [row for row in labels if str(row.get(args.split_col, "") or "").strip().lower() == args.train_split_value]
    val_rows = [row for row in labels if str(row.get(args.split_col, "") or "").strip().lower() == args.val_split_value]
    if not train_rows or not val_rows:
        raise RuntimeError(f"Invalid split counts for {args.split_col}: train={len(train_rows)} val={len(val_rows)}")

    train_pred_paths = _prediction_paths(args.type_train_predictions, args.type_train_predictions_glob)
    train_pred_by_id = _load_type_predictions(train_pred_paths, args.id_col)
    val_pred_by_id = _load_type_predictions([args.type_val_predictions], args.id_col)
    _ensure_prediction_coverage(train_rows, train_pred_by_id, args.id_col, "train", args.allow_missing_type_predictions)
    _ensure_prediction_coverage(val_rows, val_pred_by_id, args.id_col, "val", args.allow_missing_type_predictions)

    geometry_cols = parse_list(args.geometry_cols, DEFAULT_GEOMETRY_COLS)
    if args.feature_contract == "deployable_image":
        blocked = {"GEOID", "GEOID_missing", "classification_source", "estimated_units", "estimated_units_missing"}
        bad_cols = [c for c in geometry_cols if c in blocked or "geoid" in c.lower() or "source" in c.lower()]
        if bad_cols:
            raise ValueError(
                "--feature_contract deployable_image forbids non-deployable geometry/source columns: "
                + ",".join(bad_cols)
            )
    alphas = _parse_alphas(args.ridge_alphas)
    y_train = _target_log(train_rows)
    type_feature_class_names = _active_class_names(args.type_feature_policy)
    type_feature_prob_columns = [f"prob_{name}" for name in type_feature_class_names]

    train_prior = np.ones(len(PROB_COLUMNS), dtype=np.float64) / len(PROB_COLUMNS)
    if train_pred_by_id:
        raw_probs = []
        for pred in train_pred_by_id.values():
            p = np.asarray([safe_float(pred.get(col), 0.0) for col in PROB_COLUMNS], dtype=np.float64)
            if p.sum() > 0:
                raw_probs.append(p / p.sum())
        if raw_probs:
            train_prior = np.mean(np.vstack(raw_probs), axis=0)
            train_prior = train_prior / train_prior.sum()

    train_probs = _prob_matrix(train_rows, train_pred_by_id, args.id_col, train_prior, args.type_feature_policy)
    val_probs = _prob_matrix(val_rows, val_pred_by_id, args.id_col, train_prior, args.type_feature_policy)
    train_type_soft, type_soft_names = _type_probability_features(train_probs, type_feature_class_names)
    val_type_soft, _ = _type_probability_features(val_probs, type_feature_class_names)
    train_type_hard, type_hard_names = _hard_type_features(train_probs, type_feature_class_names)
    val_type_hard, _ = _hard_type_features(val_probs, type_feature_class_names)
    train_geom, geom_names = _geometry_features(train_rows, geometry_cols)
    val_geom, _ = _geometry_features(val_rows, geometry_cols)
    use_source_features = args.feature_contract == "source_aware"
    if use_source_features:
        train_source, val_source, source_names, source_categories = _source_onehot(train_rows, val_rows, args.source_col)
        base_x_train = np.column_stack([train_geom, train_source])
        base_x_val = np.column_stack([val_geom, val_source])
        base_feature_name = "geometry_source"
        soft_feature_name = "geometry_source_pred_type_soft"
        hard_feature_name = "geometry_source_pred_type_hard"
        base_names = [*geom_names, *source_names]
    else:
        source_categories = []
        base_x_train = train_geom
        base_x_val = val_geom
        base_feature_name = "geometry"
        soft_feature_name = "geometry_pred_type_soft"
        hard_feature_name = "geometry_pred_type_hard"
        base_names = list(geom_names)

    soft_x_train = np.column_stack([base_x_train, train_type_soft])
    soft_x_val = np.column_stack([base_x_val, val_type_soft])
    hard_x_train = np.column_stack([base_x_train, train_type_hard])
    hard_x_val = np.column_stack([base_x_val, val_type_hard])
    feature_sets = {
        base_feature_name: (base_x_train, base_x_val, base_names),
        soft_feature_name: (soft_x_train, soft_x_val, [*base_names, *type_soft_names]),
        hard_feature_name: (hard_x_train, hard_x_val, [*base_names, *type_hard_names]),
    }
    if not args.no_oracle:
        train_true_type, oracle_names = _true_type_features(train_rows, type_feature_class_names)
        val_true_type, _ = _true_type_features(val_rows, type_feature_class_names)
        oracle_feature_name = f"{base_feature_name}_oracle_gt_type"
        feature_sets[oracle_feature_name] = (
            np.column_stack([base_x_train, train_true_type]),
            np.column_stack([base_x_val, val_true_type]),
            [*base_names, *oracle_names],
        )

    predictions: Dict[str, Tuple[np.ndarray, str, bool, Dict[str, object]]] = {}

    if use_source_features:
        pred, meta = _fit_source_aware_ridge(
            train_geom,
            val_geom,
            y_train,
            train_rows,
            val_rows,
            args.source_col,
            alphas,
            args.min_source_train_rows,
        )
        predictions["source_aware_ridge_geometry"] = (pred, "geometry_by_source", True, meta)

        pred, meta = _fit_source_aware_ridge(
            soft_x_train,
            soft_x_val,
            y_train,
            train_rows,
            val_rows,
            args.source_col,
            alphas,
            args.min_source_train_rows,
        )
        predictions["source_aware_ridge_pred_type_soft"] = (pred, "geometry_source_pred_type_soft_by_source", True, meta)

    for feature_name, (x_train, x_val, names) in feature_sets.items():
        pred, meta = _fit_ridge_cv(x_train, y_train, x_val, alphas)
        predictions[f"ridge_{feature_name}"] = (pred, feature_name, "oracle" not in feature_name, {**meta, "n_features": len(names)})

    for feature_name in (base_feature_name, soft_feature_name, hard_feature_name):
        x_train, x_val, names = feature_sets[feature_name]
        pred, meta = _fit_hgb(x_train, y_train, x_val, args.seed, args.hgb_max_iter)
        predictions[f"hgb_{feature_name}"] = (pred, feature_name, True, {**meta, "n_features": len(names)})

    for feature_name in (base_feature_name, soft_feature_name, hard_feature_name):
        x_train, x_val, names = feature_sets[feature_name]
        pred, meta = _fit_extra_trees(x_train, y_train, x_val, args.seed, args.extra_trees_estimators)
        predictions[f"extra_trees_{feature_name}"] = (pred, feature_name, True, {**meta, "n_features": len(names)})

    if use_source_features:
        baseline_name = "source_aware_ridge_geometry"
        residual_base_train = _fit_source_aware_ridge(
            train_geom,
            train_geom,
            y_train,
            train_rows,
            train_rows,
            args.source_col,
            alphas,
            args.min_source_train_rows,
        )[0]
        residual_base_val = predictions[baseline_name][0]
        residual_model_name = "residual_hgb_on_source_aware_geometry_pred_type_soft"
        residual_feature_name = "geometry_source_pred_type_soft_residual"
    else:
        baseline_name = "hgb_geometry"
        residual_base_train = _fit_ridge_cv(train_geom, y_train, train_geom, alphas)[0]
        residual_base_val = predictions["ridge_geometry"][0]
        residual_model_name = "residual_hgb_on_ridge_geometry_pred_type_soft"
        residual_feature_name = "geometry_pred_type_soft_residual"
    residual_y = y_train - residual_base_train
    residual_pred, residual_meta = _fit_hgb(soft_x_train, residual_y, soft_x_val, args.seed + 17, args.hgb_max_iter)
    predictions[residual_model_name] = (
        residual_base_val + residual_pred,
        residual_feature_name,
        True,
        residual_meta,
    )

    expert_pred, expert_meta = _fit_soft_type_experts(
        base_x_train,
        base_x_val,
        y_train,
        train_probs,
        val_probs,
        type_feature_class_names,
        alphas,
        args.min_expert_effective_weight,
    )
    predictions["soft_type_experts_ridge"] = (expert_pred, f"{base_feature_name}_soft_type_experts", True, expert_meta)

    if not args.no_oracle:
        x_train, x_val, names = feature_sets[f"{base_feature_name}_oracle_gt_type"]
        pred, meta = _fit_hgb(x_train, y_train, x_val, args.seed, args.hgb_max_iter)
        predictions["oracle_hgb_gt_type_upper_bound"] = (pred, f"{base_feature_name}_oracle_gt_type", False, {**meta, "n_features": len(names)})

    prediction_fields = [
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
        *type_feature_prob_columns,
        "tile_base",
        "GEOID",
        "classification_source",
    ]
    result_entries = []
    detail_metrics = {}
    baseline_log_mae = None
    pred_dir = args.out_dir / "predictions"
    for model_name, (val_pred_log, feature_set, deployable, meta) in predictions.items():
        rows = _prediction_rows(val_rows, val_pred_log, val_probs, type_feature_class_names, args.source_col, args.id_col)
        metrics = _metrics_for_rows(rows, include_source_group=use_source_features)
        if model_name == baseline_name:
            baseline_log_mae = float(metrics["population_log_mae"])
        detail_metrics[model_name] = {
            "feature_set": feature_set,
            "deployable": bool(deployable),
            "metrics": metrics,
            "model_meta": meta,
        }
        if args.write_prediction_csvs:
            write_csv_rows(pred_dir / f"{model_name}.csv", rows, prediction_fields)

    if baseline_log_mae is None:
        baseline_log_mae = float("nan")
    for model_name, detail in detail_metrics.items():
        metrics = detail["metrics"]
        delta_vs_baseline = float(metrics["population_log_mae"]) - baseline_log_mae
        result_entries.append(
            {
                "model": model_name,
                "feature_set": detail["feature_set"],
                "deployable": bool(detail["deployable"]),
                "baseline_model": baseline_name,
                "population_log_mae": float(metrics["population_log_mae"]),
                "delta_log_mae_vs_baseline": delta_vs_baseline,
                "delta_log_mae_vs_source_aware_geometry": delta_vs_baseline if use_source_features else None,
                "population_factor2_hit_rate": float(metrics["population_factor2_hit_rate"]),
                "population_mae": float(metrics["population_mae"]),
                "population_rmse": float(metrics["population_rmse"]),
                "population_r2": float(metrics["population_r2"]),
            }
        )
    best_by_metric = sorted(result_entries, key=lambda x: float(x["population_log_mae"]))
    result_entries.sort(key=lambda x: (not bool(x["deployable"]), float(x["population_log_mae"])))

    summary = {
        "run_name": args.run_name,
        "labels_csv": str(args.labels_csv),
        "split_col": args.split_col,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "type_train_prediction_csvs": [str(p) for p in train_pred_paths],
        "type_val_predictions": str(args.type_val_predictions),
        "type_feature_policy": args.type_feature_policy,
        "type_feature_class_names": type_feature_class_names,
        "type_feature_prob_columns": type_feature_prob_columns,
        "geometry_cols": geometry_cols,
        "source_col": args.source_col,
        "source_categories": source_categories,
        "feature_contract": args.feature_contract,
        "disallowed_feature_columns_excluded": (
            ["classification_source", "GEOID", "GEOID_missing", "estimated_units", "estimated_units_missing"]
            if args.feature_contract == "deployable_image"
            else []
        ),
        "feature_sets": {name: fields for name, (_, _, fields) in feature_sets.items()},
        "results": result_entries,
        "details": detail_metrics,
        "best_deployable": next((row for row in result_entries if row["deployable"]), None),
        "best_overall_by_metric": best_by_metric[0] if best_by_metric else None,
    }
    save_json(args.out_dir / f"{args.summary_basename}.json", summary)
    _write_summary_csv(args.out_dir / f"{args.summary_basename}.csv", result_entries)
    print("[done] train_rows=", len(train_rows), "val_rows=", len(val_rows))
    print("[done] best_deployable=", summary["best_deployable"])
    print("[done] wrote=", args.out_dir / f"{args.summary_basename}.json")


if __name__ == "__main__":
    main()
