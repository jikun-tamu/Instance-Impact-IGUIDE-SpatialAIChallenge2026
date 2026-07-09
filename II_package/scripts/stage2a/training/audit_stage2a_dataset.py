#!/usr/bin/env python3
"""Audit Stage-2a manifest integrity, target distribution, and leakage risks."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np

try:
    from scripts.stage2a.common import (
        extract_geometry_features,
        read_csv_rows,
        resolve_path,
        safe_float,
        save_json,
        split_has_leakage,
        tile_train_val_test_split,
    )
except ImportError:  # pragma: no cover
    from II_package.scripts.stage2a.common import (
        extract_geometry_features,
        read_csv_rows,
        resolve_path,
        safe_float,
        save_json,
        split_has_leakage,
        tile_train_val_test_split,
    )


def quantiles(values: Iterable[float]) -> Dict[str, float]:
    vals = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return {}
    return {f"p{int(q)}": float(np.percentile(vals, q)) for q in [0, 25, 50, 75, 90, 95, 99, 100]}


def _missing_counts(rows: List[Mapping[str, object]], cols: List[str]) -> Dict[str, int]:
    return {c: sum(1 for r in rows if r.get(c, "") in ("", None)) for c in cols}


def audit_rows(rows: List[Mapping[str, object]], base_dir: Path | str = Path("."), sample_masks: int = 2000) -> Dict[str, object]:
    base_dir = Path(base_dir)
    missing_files = {"crop_path": 0, "mask_path": 0}
    for row in rows:
        for col in missing_files:
            p = str(row.get(col, ""))
            if not p or not resolve_path(p, base_dir).exists():
                missing_files[col] += 1

    sampled = rows[: max(0, min(sample_masks, len(rows)))]
    zero_area = 0
    geom_values: Dict[str, List[float]] = defaultdict(list)
    for row in sampled:
        mask_path = str(row.get("mask_path", ""))
        if mask_path and resolve_path(mask_path, base_dir).exists():
            geom = extract_geometry_features(resolve_path(mask_path, base_dir), row)
        else:
            geom = extract_geometry_features(None, row)
        if geom.get("mask_area_px", 0.0) <= 0:
            zero_area += 1
        for key, value in geom.items():
            geom_values[key].append(float(value))

    pops = [safe_float(r.get("estimated_population"), 0.0) for r in rows]
    units = [safe_float(r.get("estimated_units"), 0.0) for r in rows]
    pop_arr = np.asarray(pops, dtype=np.float64)
    report: Dict[str, object] = {
        "rows": len(rows),
        "tiles": len({str(r.get("tile_base", "")) for r in rows if str(r.get("tile_base", ""))}),
        "class_counts": dict(Counter(str(r.get("type_class_name", "")) for r in rows)),
        "type_class_counts": dict(Counter(str(r.get("type_class", "")) for r in rows)),
        "classification_source_counts": dict(Counter(str(r.get("classification_source", "")) for r in rows)),
        "geoid_counts": {
            "nonempty": sum(1 for r in rows if str(r.get("GEOID", ""))),
            "empty": sum(1 for r in rows if not str(r.get("GEOID", ""))),
            "unique_nonempty": len({str(r.get("GEOID", "")) for r in rows if str(r.get("GEOID", ""))}),
        },
        "population_quantiles": quantiles(pops),
        "population_outlier_counts": {
            "eq_0": int(np.sum(pop_arr == 0)) if pop_arr.size else 0,
            "gt_p95": int(np.sum(pop_arr > np.percentile(pop_arr, 95))) if pop_arr.size else 0,
            "gt_p99": int(np.sum(pop_arr > np.percentile(pop_arr, 99))) if pop_arr.size else 0,
            "gt_10000": int(np.sum(pop_arr > 10000)) if pop_arr.size else 0,
        },
        "estimated_units_quantiles": quantiles(units),
        "geometry_quantiles": {key: quantiles(vals) for key, vals in geom_values.items()},
        "missing_files": missing_files,
        "zero_area_masks_sampled": zero_area,
        "missing_label_counts": _missing_counts(
            rows,
            ["type_class", "type_class_name", "estimated_population", "footprint_m2", "GEOID", "classification_source"],
        ),
        "tile_split_leakage": False,
        "geoid_split_leakage": False,
    }
    if len(rows) >= 3:
        try:
            tile_split = tile_train_val_test_split(rows, tile_col="tile_base")
            report["tile_split_leakage"] = split_has_leakage(rows, tile_split, "tile_base")
        except Exception as exc:
            report["tile_split_error"] = str(exc)
        try:
            geoid_rows = [r for r in rows if str(r.get("GEOID", ""))]
            if len({str(r.get("GEOID", "")) for r in geoid_rows}) >= 3:
                geoid_split = tile_train_val_test_split(geoid_rows, tile_col="GEOID")
                report["geoid_split_leakage"] = split_has_leakage(geoid_rows, geoid_split, "GEOID")
        except Exception as exc:
            report["geoid_split_error"] = str(exc)
    return report


def write_markdown(report: Mapping[str, object], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage 2a Dataset Audit",
        "",
        f"- rows: `{report.get('rows')}`",
        f"- tiles: `{report.get('tiles')}`",
        f"- zero-area masks sampled: `{report.get('zero_area_masks_sampled')}`",
        f"- missing files: `{report.get('missing_files')}`",
        f"- tile split leakage: `{report.get('tile_split_leakage')}`",
        f"- GEOID split leakage: `{report.get('geoid_split_leakage')}`",
        "",
        "## Class Counts",
        "",
    ]
    for key, value in sorted(dict(report.get("class_counts", {})).items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Population Quantiles", ""])
    for key, value in dict(report.get("population_quantiles", {})).items():
        lines.append(f"- `{key}`: {value:.4f}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit Stage-2a dataset manifest.")
    p.add_argument("--labels_csv", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--out_md", type=Path, default=None)
    p.add_argument("--sample_masks", type=int, default=2000)
    p.add_argument("--base_dir", type=Path, default=Path("."))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv_rows(args.labels_csv)
    report = audit_rows(rows, base_dir=args.base_dir, sample_masks=args.sample_masks)
    save_json(args.out_json, report)
    if args.out_md:
        write_markdown(report, args.out_md)
    print("[done] rows=", report["rows"])
    print("[done] wrote=", args.out_json)


if __name__ == "__main__":
    main()
