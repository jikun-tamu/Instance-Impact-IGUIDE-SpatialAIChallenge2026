#!/usr/bin/env python3
"""Blend neural Stage-2a population predictions with a geometry ridge model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

try:
    from scripts.stage2a.common import (
        DEFAULT_GEOMETRY_COLS,
        PROB_COLUMNS,
        geometry_vector,
        load_json,
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
        DEFAULT_GEOMETRY_COLS,
        PROB_COLUMNS,
        geometry_vector,
        load_json,
        metrics_from_prediction_rows,
        parse_list,
        read_csv_rows,
        safe_float,
        save_json,
        stage2a_prediction_fields,
        write_csv_rows,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Blend Stage-2a neural population predictions with geometry ridge predictions.")
    p.add_argument("--labels_csv", type=Path, required=True)
    p.add_argument("--split_manifest", type=Path, required=True)
    p.add_argument("--val_predictions", type=Path, required=True)
    p.add_argument("--test_predictions", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--geometry_cols", type=str, default=",".join(DEFAULT_GEOMETRY_COLS))
    p.add_argument("--ridge_lambda", type=float, default=1.0)
    p.add_argument("--blend_grid_steps", type=int, default=101)
    p.add_argument(
        "--selection_metric",
        choices=["population_log_mae", "population_factor2_hit_rate"],
        default="population_log_mae",
        help="Metric optimized on validation predictions. log MAE is minimized; factor2 is maximized.",
    )
    return p.parse_args()


def _row_index(rows: Sequence[Mapping[str, object]]) -> Dict[str, Mapping[str, object]]:
    return {str(r.get("building_uid", "")): r for r in rows if str(r.get("building_uid", ""))}


def _features(rows: Sequence[Mapping[str, object]], indices: Sequence[int], cols: Sequence[str]) -> np.ndarray:
    return np.asarray([geometry_vector(rows[i], cols) for i in indices], dtype=np.float64)


def _targets(rows: Sequence[Mapping[str, object]], indices: Sequence[int]) -> np.ndarray:
    return np.asarray([np.log1p(safe_float(rows[i].get("estimated_population"), 0.0)) for i in indices], dtype=np.float64)


def _standardize(train_x: np.ndarray, other_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0, keepdims=True)
    sd = train_x.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return (train_x - mu) / sd, (other_x - mu) / sd, mu.squeeze(0), sd.squeeze(0)


def _ridge_fit(train_x: np.ndarray, train_y: np.ndarray, lam: float) -> np.ndarray:
    x = np.concatenate([np.ones((train_x.shape[0], 1)), train_x], axis=1)
    reg = float(lam) * np.eye(x.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.pinv(x.T @ x + reg) @ x.T @ train_y


def _ridge_predict(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    return x @ beta


def _prediction_vectors(pred_rows: Sequence[Mapping[str, object]], label_by_id: Mapping[str, Mapping[str, object]]) -> Tuple[np.ndarray, np.ndarray, List[Mapping[str, object]]]:
    kept: List[Mapping[str, object]] = []
    neural = []
    target = []
    for pred in pred_rows:
        bid = str(pred.get("building_uid", ""))
        label = label_by_id.get(bid)
        if label is None:
            continue
        kept.append(pred)
        neural.append(safe_float(pred.get("pred_log1p_population"), 0.0))
        target.append(np.log1p(safe_float(label.get("estimated_population"), 0.0)))
    return np.asarray(neural, dtype=np.float64), np.asarray(target, dtype=np.float64), kept


def _geometry_predictions_for_predictions(
    pred_rows: Sequence[Mapping[str, object]],
    label_by_id: Mapping[str, Mapping[str, object]],
    cols: Sequence[str],
    mu: np.ndarray,
    sd: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    xs = []
    for pred in pred_rows:
        label = label_by_id[str(pred.get("building_uid", ""))]
        xs.append(geometry_vector(label, cols))
    x = (np.asarray(xs, dtype=np.float64) - mu.reshape(1, -1)) / sd.reshape(1, -1)
    return _ridge_predict(x, beta)


def _metric_from_logs(target_log: np.ndarray, pred_log: np.ndarray, metric: str) -> float:
    if metric == "population_factor2_hit_rate":
        return float(np.mean(np.abs(target_log - pred_log) <= np.log(2.0)))
    return float(np.mean(np.abs(target_log - pred_log)))


def _select_blend_weight(neural_log: np.ndarray, geometry_log: np.ndarray, target_log: np.ndarray, steps: int, metric: str) -> Dict[str, float]:
    best = {"weight_neural": 0.0, "weight_geometry": 1.0, "validation_metric": float("inf")}
    if metric == "population_factor2_hit_rate":
        best["validation_metric"] = -float("inf")
    for w in np.linspace(0.0, 1.0, max(2, steps)):
        pred = float(w) * neural_log + (1.0 - float(w)) * geometry_log
        score = _metric_from_logs(target_log, pred, metric)
        better = score > best["validation_metric"] if metric == "population_factor2_hit_rate" else score < best["validation_metric"]
        if better:
            best = {"weight_neural": float(w), "weight_geometry": float(1.0 - w), "validation_metric": float(score)}
    return best


def _make_blended_rows(
    neural_rows: Sequence[Mapping[str, object]],
    label_by_id: Mapping[str, Mapping[str, object]],
    neural_log: np.ndarray,
    geometry_log: np.ndarray,
    weight_neural: float,
) -> List[Dict[str, object]]:
    out = []
    w = float(weight_neural)
    for pred, n_log, g_log in zip(neural_rows, neural_log, geometry_log):
        bid = str(pred.get("building_uid", ""))
        label = label_by_id[bid]
        b_log = w * float(n_log) + (1.0 - w) * float(g_log)
        row = dict(pred)
        row["pred_log1p_population"] = float(b_log)
        row["pred_population"] = float(max(0.0, np.expm1(b_log)))
        row["neural_pred_log1p_population"] = float(n_log)
        row["geometry_pred_log1p_population"] = float(g_log)
        row["true_population"] = safe_float(label.get("estimated_population"), 0.0)
        row["true_log1p_population"] = float(np.log1p(row["true_population"]))
        row["true_type_idx"] = label.get("type_class", row.get("true_type_idx", ""))
        row["true_type_class"] = label.get("type_class_name", row.get("true_type_class", ""))
        for col in PROB_COLUMNS:
            row[col] = safe_float(row.get(col), 0.0)
        out.append(row)
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = read_csv_rows(args.labels_csv)
    label_by_id = _row_index(labels)
    manifest = load_json(args.split_manifest)
    train_idx = [int(i) for i in manifest["indices"]["train"]]
    geometry_cols = parse_list(args.geometry_cols, DEFAULT_GEOMETRY_COLS)

    train_x = _features(labels, train_idx, geometry_cols)
    train_y = _targets(labels, train_idx)
    train_xs, _, mu, sd = _standardize(train_x, train_x)
    beta = _ridge_fit(train_xs, train_y, args.ridge_lambda)

    val_neural_log, val_target_log, val_rows = _prediction_vectors(read_csv_rows(args.val_predictions), label_by_id)
    test_neural_log, _, test_rows = _prediction_vectors(read_csv_rows(args.test_predictions), label_by_id)
    val_geometry_log = _geometry_predictions_for_predictions(val_rows, label_by_id, geometry_cols, mu, sd, beta)
    test_geometry_log = _geometry_predictions_for_predictions(test_rows, label_by_id, geometry_cols, mu, sd, beta)

    selected = _select_blend_weight(val_neural_log, val_geometry_log, val_target_log, args.blend_grid_steps, args.selection_metric)
    blended_test = _make_blended_rows(test_rows, label_by_id, test_neural_log, test_geometry_log, selected["weight_neural"])
    geometry_test = _make_blended_rows(test_rows, label_by_id, test_neural_log, test_geometry_log, 0.0)
    neural_test = _make_blended_rows(test_rows, label_by_id, test_neural_log, test_geometry_log, 1.0)

    extra_fields = ["neural_pred_log1p_population", "geometry_pred_log1p_population"]
    fields = [*stage2a_prediction_fields(True), *extra_fields]
    write_csv_rows(args.out_dir / "test_predictions_blended.csv", blended_test, fields)
    write_csv_rows(args.out_dir / "test_predictions_geometry_population.csv", geometry_test, fields)
    metrics = {
        "selection_metric": args.selection_metric,
        "selected_blend": selected,
        "neural_test_metrics": metrics_from_prediction_rows(neural_test),
        "geometry_population_test_metrics": metrics_from_prediction_rows(geometry_test),
        "blended_test_metrics": metrics_from_prediction_rows(blended_test),
    }
    save_json(args.out_dir / "blend_metrics.json", metrics)
    save_json(
        args.out_dir / "geometry_ridge_model.json",
        {
            "geometry_cols": geometry_cols,
            "ridge_lambda": args.ridge_lambda,
            "feature_mean": mu.tolist(),
            "feature_std": sd.tolist(),
            "beta": beta.tolist(),
            "blend": selected,
        },
    )
    print("[done] val_rows=", len(val_rows), "test_rows=", len(test_rows))
    print("[done] selected=", selected)
    print("[done] wrote=", args.out_dir / "blend_metrics.json")


if __name__ == "__main__":
    main()
