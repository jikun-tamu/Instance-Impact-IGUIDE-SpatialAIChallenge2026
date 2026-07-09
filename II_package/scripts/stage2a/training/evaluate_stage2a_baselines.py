#!/usr/bin/env python3
"""Non-deep Stage-2a baselines for the accuracy track."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

try:
    from scripts.stage2a.common import (
        CLASS_NAMES,
        DEFAULT_GEOMETRY_COLS,
        CLASS_TO_IDX,
        geometry_vector,
        load_json,
        metrics_from_prediction_rows,
        parse_list,
        read_csv_rows,
        safe_float,
        safe_int,
        save_json,
        stage2a_prediction_fields,
        write_csv_rows,
    )
except ImportError:  # pragma: no cover
    from II_package.scripts.stage2a.common import (
        CLASS_NAMES,
        DEFAULT_GEOMETRY_COLS,
        CLASS_TO_IDX,
        geometry_vector,
        load_json,
        metrics_from_prediction_rows,
        parse_list,
        read_csv_rows,
        safe_float,
        safe_int,
        save_json,
        stage2a_prediction_fields,
        write_csv_rows,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Stage-2a non-deep baselines.")
    p.add_argument("--labels_csv", type=Path, required=True)
    p.add_argument("--split_manifest", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--geometry_cols", type=str, default=",".join(DEFAULT_GEOMETRY_COLS))
    p.add_argument("--ridge_lambda", type=float, default=1.0)
    return p.parse_args()


def _targets(rows: Sequence[Mapping[str, object]], idxs: Sequence[int]):
    y_pop = np.asarray([safe_float(rows[i].get("estimated_population"), 0.0) for i in idxs], dtype=np.float64)
    y_type = np.asarray([safe_int(rows[i].get("type_class"), -1) for i in idxs], dtype=np.int64)
    return y_pop, y_type


def _features(rows: Sequence[Mapping[str, object]], idxs: Sequence[int], cols: Sequence[str], include_type: bool = False):
    xs = [geometry_vector(rows[i], cols) for i in idxs]
    x = np.asarray(xs, dtype=np.float64)
    if include_type:
        onehot = np.zeros((len(idxs), len(CLASS_NAMES)), dtype=np.float64)
        for j, i in enumerate(idxs):
            y = safe_int(rows[i].get("type_class"), -1)
            if 0 <= y < len(CLASS_NAMES):
                onehot[j, y] = 1.0
        x = np.concatenate([x, onehot], axis=1)
    return x


def _standardize(train_x: np.ndarray, test_x: np.ndarray):
    mu = train_x.mean(axis=0, keepdims=True)
    sd = train_x.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return (train_x - mu) / sd, (test_x - mu) / sd


def _ridge_fit_predict(train_x: np.ndarray, train_y_log: np.ndarray, test_x: np.ndarray, lam: float) -> np.ndarray:
    train_x = np.concatenate([np.ones((train_x.shape[0], 1)), train_x], axis=1)
    test_x = np.concatenate([np.ones((test_x.shape[0], 1)), test_x], axis=1)
    reg = float(lam) * np.eye(train_x.shape[1])
    reg[0, 0] = 0.0
    beta = np.linalg.pinv(train_x.T @ train_x + reg) @ train_x.T @ train_y_log
    return test_x @ beta


def _nearest_centroid_type(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    centroids = []
    for c in range(len(CLASS_NAMES)):
        mask = train_y == c
        if np.any(mask):
            centroids.append(train_x[mask].mean(axis=0))
        else:
            centroids.append(np.zeros(train_x.shape[1], dtype=np.float64))
    centroids = np.stack(centroids)
    d = ((test_x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1)


def _rows_for_prediction(rows, idxs, pred_log, pred_type):
    out = []
    for i, lp, ty in zip(idxs, pred_log, pred_type):
        p = np.zeros(len(CLASS_NAMES), dtype=np.float64)
        if 0 <= int(ty) < len(CLASS_NAMES):
            p[int(ty)] = 1.0
        else:
            p[:] = 1.0 / len(CLASS_NAMES)
        true_type = safe_int(rows[i].get("type_class"), -1)
        true_pop = safe_float(rows[i].get("estimated_population"), 0.0)
        pred_type_idx = int(np.argmax(p))
        out.append(
            {
                "building_uid": rows[i].get("building_uid", ""),
                "pred_population": float(max(0.0, np.expm1(lp))),
                "pred_log1p_population": float(lp),
                "pred_type_idx": pred_type_idx,
                "pred_type_class": CLASS_NAMES[pred_type_idx],
                "pred_type_conf": float(p[pred_type_idx]),
                **{f"prob_{name}": float(p[j]) for j, name in enumerate(CLASS_NAMES)},
                "crop_path": rows[i].get("crop_path", ""),
                "mask_path": rows[i].get("mask_path", ""),
                "tile_base": rows[i].get("tile_base", ""),
                "GEOID": rows[i].get("GEOID", ""),
                "classification_source": rows[i].get("classification_source", ""),
                "true_population": true_pop,
                "true_log1p_population": float(np.log1p(true_pop)),
                "true_type_idx": true_type,
                "true_type_class": CLASS_NAMES[true_type] if 0 <= true_type < len(CLASS_NAMES) else "",
            }
        )
    return out


def _median_by_type(train_rows, train_idx):
    med = {}
    y_global = []
    for i in train_idx:
        y = safe_float(train_rows[i].get("estimated_population"), 0.0)
        y_global.append(np.log1p(y))
    global_med = float(np.median(y_global)) if y_global else 0.0
    for c in range(len(CLASS_NAMES)):
        vals = [np.log1p(safe_float(train_rows[i].get("estimated_population"), 0.0)) for i in train_idx if safe_int(train_rows[i].get("type_class"), -1) == c]
        med[c] = float(np.median(vals)) if vals else global_med
    return global_med, med


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv_rows(args.labels_csv)
    manifest = load_json(args.split_manifest)
    train_idx = list(map(int, manifest["indices"]["train"]))
    test_idx = list(map(int, manifest["indices"]["test"]))
    geom_cols = parse_list(args.geometry_cols, DEFAULT_GEOMETRY_COLS)
    train_y_pop, train_y_type = _targets(rows, train_idx)
    test_y_pop, test_y_type = _targets(rows, test_idx)
    global_med, type_med = _median_by_type(rows, train_idx)

    baselines: Dict[str, List[Dict[str, object]]] = {}
    baselines["global_median_population"] = _rows_for_prediction(rows, test_idx, np.full(len(test_idx), global_med), np.full(len(test_idx), 2))
    baselines["median_by_gt_type"] = _rows_for_prediction(rows, test_idx, np.asarray([type_med.get(int(y), global_med) for y in test_y_type]), test_y_type)

    train_x = _features(rows, train_idx, geom_cols, include_type=False)
    test_x = _features(rows, test_idx, geom_cols, include_type=False)
    train_xs, test_xs = _standardize(train_x, test_x)
    pred_type_geom = _nearest_centroid_type(train_xs, train_y_type, test_xs)
    baselines["median_by_predicted_oof_type"] = _rows_for_prediction(
        rows,
        test_idx,
        np.asarray([type_med.get(int(y), global_med) for y in pred_type_geom]),
        pred_type_geom,
    )
    pred_geom_log = _ridge_fit_predict(train_xs, np.log1p(train_y_pop), test_xs, args.ridge_lambda)
    baselines["geometry_only_regressor"] = _rows_for_prediction(rows, test_idx, pred_geom_log, pred_type_geom)

    train_xt = _features(rows, train_idx, geom_cols, include_type=True)
    test_xt = _features(rows, test_idx, geom_cols, include_type=True)
    train_xts, test_xts = _standardize(train_xt, test_xt)
    pred_oracle_log = _ridge_fit_predict(train_xts, np.log1p(train_y_pop), test_xts, args.ridge_lambda)
    baselines["oracle_gt_type_geometry"] = _rows_for_prediction(rows, test_idx, pred_oracle_log, test_y_type)

    summary = {}
    for name, pred_rows in baselines.items():
        out_csv = args.out_dir / f"{name}.csv"
        out_json = args.out_dir / f"{name}_metrics.json"
        metrics = metrics_from_prediction_rows(pred_rows)
        write_csv_rows(out_csv, pred_rows, stage2a_prediction_fields(True))
        save_json(out_json, metrics)
        summary[name] = metrics
    save_json(args.out_dir / "baseline_summary.json", summary)
    print("[done] baselines=", ", ".join(sorted(summary)))
    print("[done] wrote=", args.out_dir / "baseline_summary.json")


if __name__ == "__main__":
    main()
