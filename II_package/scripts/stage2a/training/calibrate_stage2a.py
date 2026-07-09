#!/usr/bin/env python3
"""Calibrate Stage-2a type probabilities and summarize population errors."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

try:
    from scripts.stage2a.common import (
        CLASS_NAMES,
        PROB_COLUMNS,
        classification_metrics,
        grouped_regression_metrics,
        metrics_from_prediction_rows,
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
        PROB_COLUMNS,
        classification_metrics,
        grouped_regression_metrics,
        metrics_from_prediction_rows,
        read_csv_rows,
        safe_float,
        safe_int,
        save_json,
        stage2a_prediction_fields,
        write_csv_rows,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate Stage-2a type probabilities.")
    p.add_argument("--val_predictions", type=Path, required=True)
    p.add_argument("--test_predictions", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--temperature_min", type=float, default=0.5)
    p.add_argument("--temperature_max", type=float, default=5.0)
    p.add_argument("--temperature_steps", type=int, default=200)
    return p.parse_args()


def _probs(rows):
    return np.asarray([[safe_float(r.get(c), 0.0) for c in PROB_COLUMNS] for r in rows], dtype=np.float64)


def _labels(rows):
    return np.asarray([safe_int(r.get("true_type_idx"), -1) for r in rows], dtype=np.int64)


def _temperature_scale(probs: np.ndarray, temp: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / float(temp)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def _nll(probs: np.ndarray, y: np.ndarray) -> float:
    valid = y >= 0
    if not np.any(valid):
        return 0.0
    p = np.clip(probs[valid, y[valid]], 1e-12, 1.0)
    return float(-np.log(p).mean())


def fit_temperature(rows, t_min: float, t_max: float, steps: int) -> float:
    probs = _probs(rows)
    y = _labels(rows)
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(t_min, t_max, max(2, steps)):
        nll = _nll(_temperature_scale(probs, float(t)), y)
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def apply_temperature(rows, temperature: float):
    probs = _temperature_scale(_probs(rows), temperature)
    out = []
    for row, p in zip(rows, probs):
        r = dict(row)
        pred = int(np.argmax(p))
        r["pred_type_idx"] = pred
        r["pred_type_class"] = CLASS_NAMES[pred]
        r["pred_type_conf"] = float(p[pred])
        for col, val in zip(PROB_COLUMNS, p):
            r[col] = float(val)
        out.append(r)
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    val_rows = read_csv_rows(args.val_predictions)
    test_rows = read_csv_rows(args.test_predictions)
    temp = fit_temperature(val_rows, args.temperature_min, args.temperature_max, args.temperature_steps)
    cal_test = apply_temperature(test_rows, temp)
    raw_metrics = metrics_from_prediction_rows(test_rows)
    cal_metrics = metrics_from_prediction_rows(cal_test)
    save_json(
        args.out_dir / "calibration_metrics.json",
        {
            "temperature": temp,
            "raw_type_nll": raw_metrics.get("type_nll"),
            "raw_type_ece": raw_metrics.get("type_ece"),
            "temperature_type_nll": cal_metrics.get("type_nll"),
            "temperature_type_ece": cal_metrics.get("type_ece"),
            "raw_type_macro_f1": raw_metrics.get("type_macro_f1"),
            "temperature_type_macro_f1": cal_metrics.get("type_macro_f1"),
        },
    )
    write_csv_rows(args.out_dir / "test_predictions_calibrated.csv", cal_test, stage2a_prediction_fields(True))
    save_json(args.out_dir / "population_error_by_type.json", grouped_regression_metrics(cal_test, "true_type_class"))
    save_json(args.out_dir / "population_error_by_bin.json", cal_metrics.get("population_by_bin", {}))
    print("[done] temperature=", round(temp, 4))
    print("[done] wrote=", args.out_dir / "calibration_metrics.json")


if __name__ == "__main__":
    main()
