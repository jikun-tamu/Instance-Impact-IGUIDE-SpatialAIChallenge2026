#!/usr/bin/env python3
"""Ensemble Stage-2a prediction CSVs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    from scripts.stage2a.common import CLASS_NAMES, PROB_COLUMNS, metrics_from_prediction_rows, read_csv_rows, safe_float, save_json, write_csv_rows
except ImportError:  # pragma: no cover
    from II_package.scripts.stage2a.common import CLASS_NAMES, PROB_COLUMNS, metrics_from_prediction_rows, read_csv_rows, safe_float, save_json, write_csv_rows


OUT_FIELDS = [
    "building_uid",
    "pred_population_ensemble",
    "pred_log1p_population_ensemble",
    "pred_population_std",
    "pred_type_idx_ensemble",
    "pred_type_class_ensemble",
    "pred_type_conf_ensemble",
    "pred_type_entropy",
    *PROB_COLUMNS,
    "true_population",
    "true_log1p_population",
    "true_type_idx",
    "true_type_class",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensemble Stage-2a predictions.")
    p.add_argument("--prediction_csvs", type=str, required=True)
    p.add_argument("--weights", type=str, default="")
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--out_metrics", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(x.strip()) for x in args.prediction_csvs.split(",") if x.strip()]
    if not paths:
        raise ValueError("No prediction CSVs provided")
    weights = np.asarray([float(x) for x in args.weights.split(",") if x.strip()] or [1.0] * len(paths), dtype=np.float64)
    if len(weights) != len(paths):
        raise ValueError("--weights length must match --prediction_csvs")
    weights = weights / weights.sum()
    tables = [read_csv_rows(p) for p in paths]
    by_id: List[Dict[str, Dict[str, str]]] = [{r["building_uid"]: r for r in table} for table in tables]
    common_ids = sorted(set.intersection(*(set(t) for t in by_id)))
    out_rows = []
    metric_rows = []
    for bid in common_ids:
        rows = [t[bid] for t in by_id]
        pop_logs = np.asarray([safe_float(r.get("pred_log1p_population"), 0.0) for r in rows], dtype=np.float64)
        pops = np.maximum(0.0, np.expm1(pop_logs))
        probs = np.asarray([[safe_float(r.get(c), 0.0) for c in PROB_COLUMNS] for r in rows], dtype=np.float64)
        prob = np.sum(probs * weights.reshape(-1, 1), axis=0)
        prob = prob / np.clip(prob.sum(), 1e-12, None)
        pop_log = float(np.sum(pop_logs * weights))
        pred_idx = int(np.argmax(prob))
        entropy = float(-np.sum(prob * np.log(np.clip(prob, 1e-12, 1.0))))
        base = rows[0]
        row = {
            "building_uid": bid,
            "pred_population_ensemble": float(np.expm1(pop_log)),
            "pred_log1p_population_ensemble": pop_log,
            "pred_population_std": float(np.sqrt(np.sum(((pops - np.sum(pops * weights)) ** 2) * weights))),
            "pred_type_idx_ensemble": pred_idx,
            "pred_type_class_ensemble": CLASS_NAMES[pred_idx],
            "pred_type_conf_ensemble": float(prob[pred_idx]),
            "pred_type_entropy": entropy,
            "true_population": base.get("true_population", ""),
            "true_log1p_population": base.get("true_log1p_population", ""),
            "true_type_idx": base.get("true_type_idx", ""),
            "true_type_class": base.get("true_type_class", ""),
        }
        for col, val in zip(PROB_COLUMNS, prob):
            row[col] = float(val)
        out_rows.append(row)
        metric_row = {
            "building_uid": bid,
            "pred_population": row["pred_population_ensemble"],
            "pred_log1p_population": row["pred_log1p_population_ensemble"],
            "pred_type_idx": row["pred_type_idx_ensemble"],
            "pred_type_class": row["pred_type_class_ensemble"],
            "pred_type_conf": row["pred_type_conf_ensemble"],
            **{col: row[col] for col in PROB_COLUMNS},
            "true_population": row["true_population"],
            "true_log1p_population": row["true_log1p_population"],
            "true_type_idx": row["true_type_idx"],
            "true_type_class": row["true_type_class"],
        }
        metric_rows.append(metric_row)
    write_csv_rows(args.out_csv, out_rows, OUT_FIELDS)
    save_json(args.out_metrics, metrics_from_prediction_rows(metric_rows))
    print("[done] rows=", len(out_rows))
    print("[done] wrote=", args.out_csv)


if __name__ == "__main__":
    main()
