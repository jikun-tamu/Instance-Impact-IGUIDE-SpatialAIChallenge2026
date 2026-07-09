#!/usr/bin/env python3
"""Package the selected native 3-class Stage-2a population model.

This fits the deployable M33/M35-style population model:

    log_footprint_m2 + soft native 3-class type + residual HGB on ridge geometry

The resulting pickle is intended for inference-time use by
`infer_stage2a_native3_ensemble.py`.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

try:
    from scripts.stage2a.common import (
        geometry_vector,
        metrics_from_prediction_rows,
        parse_list,
        read_csv_rows,
        safe_float,
        save_json,
        stage2a_prediction_fields,
        write_csv_rows,
    )
except ImportError:  # pragma: no cover
    from II_package.scripts.stage2a.common import (
        geometry_vector,
        metrics_from_prediction_rows,
        parse_list,
        read_csv_rows,
        safe_float,
        save_json,
        stage2a_prediction_fields,
        write_csv_rows,
    )


DEFAULT_CLASS_NAMES = ["residential_small", "residential_multi", "commercial"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Package the selected Stage2a native3 population model.")
    p.add_argument("--labels_csv", type=Path, required=True)
    p.add_argument("--type_train_predictions", type=Path, required=True)
    p.add_argument("--type_val_predictions", type=Path, default=None)
    p.add_argument("--out_pkl", type=Path, required=True)
    p.add_argument("--out_summary_json", type=Path, required=True)
    p.add_argument("--out_val_predictions", type=Path, default=None)
    p.add_argument("--split_col", type=str, default="m20_split")
    p.add_argument("--train_split_value", type=str, default="train")
    p.add_argument("--val_split_value", type=str, default="val")
    p.add_argument("--id_col", type=str, default="building_uid")
    p.add_argument("--geometry_cols", type=str, default="log_footprint_m2")
    p.add_argument("--class_names", type=str, default=",".join(DEFAULT_CLASS_NAMES))
    p.add_argument("--ridge_alphas", type=str, default="0.01,0.1,1.0,10.0,100.0")
    p.add_argument("--hgb_max_iter", type=int, default=350)
    p.add_argument("--seed", type=int, default=43033)
    return p.parse_args()


def _row_id(row: Mapping[str, object], id_col: str) -> str:
    return str(row.get(id_col, "") or "").strip()


def _parse_alphas(text: str) -> List[float]:
    out = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not out:
        raise ValueError("--ridge_alphas produced no values")
    return out


def _target_log(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    return np.asarray([math.log1p(max(0.0, safe_float(row.get("estimated_population"), 0.0))) for row in rows], dtype=np.float64)


def _load_predictions(path: Path, id_col: str) -> Dict[str, Dict[str, str]]:
    by_id: Dict[str, Dict[str, str]] = {}
    duplicates = []
    for row in read_csv_rows(path):
        rid = _row_id(row, id_col)
        if not rid:
            continue
        if rid in by_id:
            duplicates.append(rid)
            continue
        by_id[rid] = dict(row)
    if duplicates:
        sample = ", ".join(sorted(set(duplicates))[:5])
        raise RuntimeError(f"Duplicate prediction ids in {path}: {sample}")
    return by_id


def _class_prob_matrix(
    rows: Sequence[Mapping[str, object]],
    pred_by_id: Mapping[str, Mapping[str, object]],
    class_names: Sequence[str],
    id_col: str,
) -> np.ndarray:
    probs = []
    missing = []
    cols = [f"prob_{name}" for name in class_names]
    for row in rows:
        rid = _row_id(row, id_col)
        pred = pred_by_id.get(rid)
        if pred is None:
            missing.append(rid)
            continue
        p = np.asarray([safe_float(pred.get(col), 0.0) for col in cols], dtype=np.float64)
        total = float(p.sum())
        if total <= 0:
            idx = int(safe_float(pred.get("pred_type_idx"), -1))
            p = np.zeros(len(class_names), dtype=np.float64)
            if 0 <= idx < len(class_names):
                p[idx] = 1.0
            else:
                p[:] = 1.0 / len(class_names)
        else:
            p = p / total
        probs.append(p)
    if missing:
        sample = ", ".join(missing[:5])
        raise RuntimeError(f"Missing {len(missing)} type predictions; examples: {sample}")
    return np.vstack(probs).astype(np.float64)


def _type_soft_features(probs: np.ndarray, class_names: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
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


def _geometry_features(rows: Sequence[Mapping[str, object]], geometry_cols: Sequence[str]) -> np.ndarray:
    return np.asarray([geometry_vector(row, geometry_cols) for row in rows], dtype=np.float64)


def _fit_ridge(train_x: np.ndarray, train_y: np.ndarray, alphas: Sequence[float]):
    from sklearn.linear_model import RidgeCV

    mu = train_x.mean(axis=0, keepdims=True)
    sd = train_x.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    xs = (train_x - mu) / sd
    cv = min(5, max(2, len(train_y) // 1000))
    model = RidgeCV(alphas=list(alphas), cv=cv)
    model.fit(xs, train_y)
    return model, mu.squeeze(0), sd.squeeze(0)


def _predict_ridge(model, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict((x - mean.reshape(1, -1)) / std.reshape(1, -1)), dtype=np.float64)


def _fit_hgb(train_x: np.ndarray, train_y: np.ndarray, seed: int, max_iter: int):
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
    except TypeError:  # pragma: no cover
        model = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=0.04, random_state=seed)
        loss = "default"
    model.fit(train_x, train_y)
    return model, loss


def _apply_artifact(artifact: Mapping[str, object], rows: Sequence[Mapping[str, object]], probs: np.ndarray) -> np.ndarray:
    geometry_cols = list(artifact["geometry_cols"])
    class_names = list(artifact["class_names"])
    geom = _geometry_features(rows, geometry_cols)
    type_x, type_names = _type_soft_features(probs, class_names)
    expected = list(artifact["type_feature_names"])
    if type_names != expected:
        raise ValueError(f"Type feature mismatch: expected={expected}, got={type_names}")
    base = _predict_ridge(artifact["ridge_model"], geom, np.asarray(artifact["ridge_mean"]), np.asarray(artifact["ridge_std"]))
    residual_x = np.column_stack([geom, type_x])
    return base + np.asarray(artifact["residual_model"].predict(residual_x), dtype=np.float64)


def _row_class_name(row: Mapping[str, object]) -> str:
    return str(row.get("type_class_name", "") or row.get("true_type_class", "") or "").strip()


def _prediction_rows(
    rows: Sequence[Mapping[str, object]],
    pred_log: np.ndarray,
    probs: np.ndarray,
    class_names: Sequence[str],
    id_col: str,
) -> List[Dict[str, object]]:
    out = []
    for i, row in enumerate(rows):
        true_pop = max(0.0, safe_float(row.get("estimated_population"), safe_float(row.get("true_population"), 0.0)))
        pred_idx = int(np.argmax(probs[i]))
        true_name = _row_class_name(row)
        true_idx = class_names.index(true_name) if true_name in class_names else -1
        item: Dict[str, object] = {
            "building_uid": _row_id(row, id_col),
            "pred_population": float(max(0.0, math.expm1(float(pred_log[i])))),
            "pred_log1p_population": float(pred_log[i]),
            "pred_type_idx": pred_idx,
            "pred_type_class": class_names[pred_idx],
            "pred_type_conf": float(np.max(probs[i])),
            "crop_path": row.get("crop_path", ""),
            "mask_path": row.get("mask_path", ""),
            "tile_base": row.get("tile_base", row.get("tile_id", "")),
            "GEOID": row.get("GEOID", ""),
            "classification_source": row.get("classification_source", ""),
            "true_population": float(true_pop),
            "true_log1p_population": float(math.log1p(true_pop)),
            "true_type_idx": true_idx,
            "true_type_class": class_names[true_idx] if 0 <= true_idx < len(class_names) else "",
        }
        for j, name in enumerate(class_names):
            item[f"prob_{name}"] = float(probs[i, j])
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    labels = read_csv_rows(args.labels_csv)
    class_names = parse_list(args.class_names, DEFAULT_CLASS_NAMES)
    geometry_cols = parse_list(args.geometry_cols, ["log_footprint_m2"])
    alphas = _parse_alphas(args.ridge_alphas)
    train_rows = [row for row in labels if str(row.get(args.split_col, "")).strip().lower() == args.train_split_value]
    val_rows = [row for row in labels if str(row.get(args.split_col, "")).strip().lower() == args.val_split_value]
    if not train_rows:
        raise RuntimeError(f"No train rows found for {args.split_col}={args.train_split_value}")

    train_pred = _load_predictions(args.type_train_predictions, args.id_col)
    train_probs = _class_prob_matrix(train_rows, train_pred, class_names, args.id_col)
    train_geom = _geometry_features(train_rows, geometry_cols)
    train_type_x, type_feature_names = _type_soft_features(train_probs, class_names)
    train_y = _target_log(train_rows)

    ridge, ridge_mean, ridge_std = _fit_ridge(train_geom, train_y, alphas)
    base_train = _predict_ridge(ridge, train_geom, ridge_mean, ridge_std)
    residual_y = train_y - base_train
    residual_x = np.column_stack([train_geom, train_type_x])
    residual_model, hgb_loss = _fit_hgb(residual_x, residual_y, args.seed + 17, args.hgb_max_iter)

    artifact = {
        "format": "stage2a_native3_population_residual_hgb_v1",
        "model_name": "log_footprint_m2_soft_native3_type_residual_hgb",
        "class_names": class_names,
        "geometry_cols": geometry_cols,
        "type_feature_names": type_feature_names,
        "ridge_model": ridge,
        "ridge_alpha": float(ridge.alpha_),
        "ridge_mean": ridge_mean,
        "ridge_std": ridge_std,
        "residual_model": residual_model,
        "residual_model_type": "HistGradientBoostingRegressor",
        "residual_loss": hgb_loss,
        "hgb_max_iter": int(args.hgb_max_iter),
        "seed": int(args.seed),
        "train_rows": len(train_rows),
        "labels_csv": str(args.labels_csv),
        "type_train_predictions": str(args.type_train_predictions),
    }

    args.out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_pkl.open("wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        "format": artifact["format"],
        "model_name": artifact["model_name"],
        "class_names": class_names,
        "geometry_cols": geometry_cols,
        "type_feature_names": type_feature_names,
        "ridge_alpha": artifact["ridge_alpha"],
        "residual_model_type": artifact["residual_model_type"],
        "residual_loss": artifact["residual_loss"],
        "hgb_max_iter": artifact["hgb_max_iter"],
        "seed": artifact["seed"],
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "labels_csv": str(args.labels_csv),
        "type_train_predictions": str(args.type_train_predictions),
        "out_pkl": str(args.out_pkl),
    }

    if args.type_val_predictions is not None:
        if not val_rows:
            raise RuntimeError(f"No val rows found for {args.split_col}={args.val_split_value}")
        val_pred = _load_predictions(args.type_val_predictions, args.id_col)
        val_probs = _class_prob_matrix(val_rows, val_pred, class_names, args.id_col)
        val_pred_log = _apply_artifact(artifact, val_rows, val_probs)
        val_out = _prediction_rows(val_rows, val_pred_log, val_probs, class_names, args.id_col)
        metrics = metrics_from_prediction_rows(val_out, class_names=class_names)
        summary["validation"] = metrics
        summary["type_val_predictions"] = str(args.type_val_predictions)
        if args.out_val_predictions is not None:
            fields = stage2a_prediction_fields(include_truth=True, class_names=class_names)
            write_csv_rows(args.out_val_predictions, val_out, fields)
            summary["out_val_predictions"] = str(args.out_val_predictions)

    save_json(args.out_summary_json, summary)
    print("[done] train_rows=", len(train_rows))
    if "validation" in summary:
        v = summary["validation"]
        print("[done] val_population_log_mae=", round(float(v["population_log_mae"]), 6))
        print("[done] val_population_factor2=", round(float(v["population_factor2_hit_rate"]), 6))
    print("[done] wrote=", args.out_pkl)
    print("[done] wrote=", args.out_summary_json)


if __name__ == "__main__":
    main()
