#!/usr/bin/env python3
"""Average Stage-2a type prediction CSVs by building id."""

from __future__ import annotations

import argparse
import glob
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

try:
    from scripts.stage2a.common import CLASS_NAMES, PROB_COLUMNS, read_csv_rows, safe_float, write_csv_rows
except ImportError:  # pragma: no cover
    from II_package.scripts.stage2a.common import CLASS_NAMES, PROB_COLUMNS, read_csv_rows, safe_float, write_csv_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Average Stage2a type prediction probabilities.")
    p.add_argument("--prediction_csv", type=Path, action="append", default=[])
    p.add_argument("--prediction_glob", type=str, action="append", default=[])
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--id_col", type=str, default="building_uid")
    p.add_argument("--expected_member_count", type=int, default=0)
    p.add_argument("--require_expected_member_count", action="store_true")
    return p.parse_args()


def prediction_paths(paths: Sequence[Path], patterns: Sequence[str]) -> List[Path]:
    out = [Path(p) for p in paths]
    for pattern in patterns:
        out.extend(Path(p) for p in sorted(glob.glob(pattern)))
    unique: List[Path] = []
    seen = set()
    for path in out:
        key = str(path)
        if path.is_file() and key not in seen:
            unique.append(path)
            seen.add(key)
    if not unique:
        raise FileNotFoundError("No prediction CSVs were provided.")
    return unique


def row_id(row: Mapping[str, object], id_col: str) -> str:
    return str(row.get(id_col, "") or "").strip()


def entropy(probs: np.ndarray) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def average_rows(rows: Sequence[Mapping[str, str]], id_col: str) -> Dict[str, object]:
    base = dict(rows[0])
    probs = np.asarray([[safe_float(row.get(col), 0.0) for col in PROB_COLUMNS] for row in rows], dtype=np.float64)
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = np.divide(probs, np.clip(row_sums, 1e-12, None))
    mean_probs = probs.mean(axis=0)
    total = float(mean_probs.sum())
    if total > 0:
        mean_probs = mean_probs / total
    pred_idx = int(np.argmax(mean_probs))
    out: Dict[str, object] = base
    out.pop("_prediction_source_csv", None)
    out["pred_type_idx"] = pred_idx
    out["pred_type_class"] = CLASS_NAMES[pred_idx]
    out["pred_type_conf"] = float(mean_probs[pred_idx])
    for col, value in zip(PROB_COLUMNS, mean_probs):
        out[col] = float(value)
    std_probs = probs.std(axis=0, ddof=0)
    for col, value in zip(PROB_COLUMNS, std_probs):
        out[f"{col}_member_std"] = float(value)
    out["ensemble_member_count"] = len(rows)
    out["ensemble_mean_entropy"] = entropy(mean_probs)
    out["ensemble_mean_prob_member_std"] = float(std_probs.mean())
    out["ensemble_prediction_sources"] = ";".join(sorted(str(row.get("_prediction_source_csv", "")) for row in rows))
    if id_col not in out:
        out[id_col] = row_id(base, id_col)
    return out


def fieldnames(rows: Sequence[Mapping[str, object]]) -> List[str]:
    preferred = [
        "building_uid",
        "pred_population",
        "pred_log1p_population",
        "pred_type_idx",
        "pred_type_class",
        "pred_type_conf",
        *PROB_COLUMNS,
        *[f"{col}_member_std" for col in PROB_COLUMNS],
        "ensemble_member_count",
        "ensemble_mean_entropy",
        "ensemble_mean_prob_member_std",
        "crop_path",
        "mask_path",
        "tile_base",
        "GEOID",
        "classification_source",
        "true_population",
        "true_log1p_population",
        "true_type_idx",
        "true_type_class",
        "ensemble_prediction_sources",
    ]
    keys = set()
    for row in rows:
        keys.update(row.keys())
    ordered = [key for key in preferred if key in keys]
    ordered.extend(sorted(key for key in keys if key not in set(ordered)))
    return ordered


def main() -> None:
    args = parse_args()
    paths = prediction_paths(args.prediction_csv, args.prediction_glob)
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for path in paths:
        for row in read_csv_rows(path):
            rid = row_id(row, args.id_col)
            if not rid:
                continue
            item = dict(row)
            item["_prediction_source_csv"] = str(path)
            grouped[rid].append(item)

    if args.require_expected_member_count and args.expected_member_count > 0:
        bad = [rid for rid, items in grouped.items() if len(items) != args.expected_member_count]
        if bad:
            sample = ", ".join(sorted(bad)[:5])
            raise RuntimeError(
                f"{len(bad)} ids do not have expected_member_count={args.expected_member_count}; examples: {sample}"
            )

    out = [average_rows(items, args.id_col) for _, items in sorted(grouped.items())]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv_rows(args.out_csv, out, fieldnames(out))
    counts = [len(items) for items in grouped.values()]
    print("[done] prediction_csvs=", len(paths))
    print("[done] rows=", len(out))
    print("[done] member_count_min=", min(counts) if counts else 0)
    print("[done] member_count_max=", max(counts) if counts else 0)
    print("[done] wrote=", args.out_csv)


if __name__ == "__main__":
    main()
