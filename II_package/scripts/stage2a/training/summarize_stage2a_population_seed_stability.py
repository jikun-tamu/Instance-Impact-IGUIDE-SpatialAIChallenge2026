#!/usr/bin/env python3
"""Summarize Stage-2a population stability across type/model seeds."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


METRICS = [
    "population_log_mae",
    "population_factor2_hit_rate",
    "population_mae",
    "population_rmse",
    "population_r2",
    "delta_log_mae_vs_baseline",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate Stage2a population summary JSONs across seeds.")
    p.add_argument("--summary_glob", type=str, action="append", required=True)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, default=None)
    p.add_argument(
        "--target_model",
        type=str,
        default="residual_hgb_on_ridge_geometry_pred_type_soft",
        help="Model name to treat as the fixed final-candidate row when present.",
    )
    p.add_argument(
        "--protocol_note",
        type=str,
        default=(
            "Population stability is computed over drop-both Stage2a runs that use OOF type predictions "
            "for training rows and validation type predictions from the matching final type seed."
        ),
    )
    return p.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, sort_keys=True)


def to_jsonable(obj: object) -> object:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def infer_seed(path: Path, summary: Mapping[str, object]) -> int | None:
    texts = [
        str(summary.get("run_name", "")),
        str(summary.get("type_val_predictions", "")),
        str(path),
    ]
    for text in texts:
        match = re.search(r"seed(\d+)", text)
        if match:
            return int(match.group(1))
    return None


def metric_stats(values: Iterable[float]) -> Dict[str, float | int | None]:
    vals = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=0)),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def expand_paths(patterns: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for pattern in patterns:
        for raw in sorted(glob.glob(pattern)):
            path = Path(raw)
            if not path.is_file():
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(path)
    if not out:
        raise FileNotFoundError("No population summary JSONs matched --summary_glob")
    return out


def geometry_signature(summary: Mapping[str, object]) -> str:
    cols = summary.get("geometry_cols")
    if isinstance(cols, list):
        return ",".join(str(c) for c in cols)
    if cols:
        return str(cols)
    return "__unknown_geometry__"


def find_result(summary: Mapping[str, object], target_model: str) -> Mapping[str, object] | None:
    for row in summary.get("results", []) or []:
        if isinstance(row, Mapping) and row.get("model") == target_model:
            return row
    return None


def row_from_summary(path: Path, summary: Mapping[str, object], target_model: str) -> Dict[str, object]:
    best = summary.get("best_deployable") if isinstance(summary.get("best_deployable"), Mapping) else {}
    target = find_result(summary, target_model) or best
    seed = infer_seed(path, summary)
    geometry = geometry_signature(summary)
    train_csvs = summary.get("type_train_prediction_csvs") or []
    return {
        "seed": seed,
        "summary_json": str(path),
        "run_name": summary.get("run_name"),
        "labels_csv": summary.get("labels_csv"),
        "split_col": summary.get("split_col"),
        "train_rows": summary.get("train_rows"),
        "val_rows": summary.get("val_rows"),
        "geometry_cols": geometry,
        "feature_contract": summary.get("feature_contract"),
        "type_feature_policy": summary.get("type_feature_policy"),
        "type_train_prediction_count": len(train_csvs) if isinstance(train_csvs, list) else None,
        "type_val_predictions": summary.get("type_val_predictions"),
        "target_model": target.get("model"),
        "target_feature_set": target.get("feature_set"),
        "target_deployable": target.get("deployable"),
        "target_population_log_mae": safe_float(target.get("population_log_mae")),
        "target_population_factor2_hit_rate": safe_float(target.get("population_factor2_hit_rate")),
        "target_population_mae": safe_float(target.get("population_mae")),
        "target_population_rmse": safe_float(target.get("population_rmse")),
        "target_population_r2": safe_float(target.get("population_r2")),
        "target_delta_log_mae_vs_baseline": safe_float(target.get("delta_log_mae_vs_baseline")),
        "best_deployable_model": best.get("model"),
        "best_deployable_feature_set": best.get("feature_set"),
        "best_deployable_population_log_mae": safe_float(best.get("population_log_mae")),
        "best_deployable_population_factor2_hit_rate": safe_float(best.get("population_factor2_hit_rate")),
        "best_deployable_population_mae": safe_float(best.get("population_mae")),
        "best_deployable_population_r2": safe_float(best.get("population_r2")),
    }


def aggregate(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("geometry_cols")), str(row.get("target_model")))].append(row)
    out: List[Dict[str, object]] = []
    for (geometry, model), items in sorted(grouped.items()):
        group: Dict[str, object] = {
            "geometry_cols": geometry,
            "target_model": model,
            "run_count": len(items),
            "seeds": [row.get("seed") for row in items],
        }
        for metric in METRICS:
            key = f"target_{metric}"
            if metric == "delta_log_mae_vs_baseline":
                key = "target_delta_log_mae_vs_baseline"
            group[metric] = metric_stats([safe_float(row.get(key)) for row in items])
        out.append(group)
    out.sort(
        key=lambda row: (
            safe_float((row.get("population_log_mae") or {}).get("mean"), float("inf")),
            str(row.get("geometry_cols")),
        )
    )
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = [
        "seed",
        "run_name",
        "geometry_cols",
        "feature_contract",
        "type_feature_policy",
        "type_train_prediction_count",
        "target_model",
        "target_feature_set",
        "target_population_log_mae",
        "target_delta_log_mae_vs_baseline",
        "target_population_factor2_hit_rate",
        "target_population_mae",
        "target_population_rmse",
        "target_population_r2",
        "best_deployable_model",
        "best_deployable_population_log_mae",
        "summary_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    paths = expand_paths(args.summary_glob)
    rows = [row_from_summary(path, load_json(path), args.target_model) for path in paths]
    payload = {
        "protocol_note": args.protocol_note,
        "summary_globs": args.summary_glob,
        "target_model": args.target_model,
        "runs": rows,
        "groups": aggregate(rows),
    }
    save_json(args.out_json, payload)
    if args.out_csv:
        write_csv(args.out_csv, rows)
    print("[done] summarized", len(rows), "population runs")
    print("[done] wrote", args.out_json)


if __name__ == "__main__":
    main()
