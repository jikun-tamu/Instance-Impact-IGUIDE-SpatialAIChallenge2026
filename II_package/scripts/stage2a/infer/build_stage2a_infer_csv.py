#!/usr/bin/env python3
"""Build a Stage-2a inference CSV from shared instance artifacts.

Input contract:
- shared_instance_samples.csv produced by scripts/shared/generate_shared_instance_subimages.py

Output contract:
- CSV with at least:
  - building_uid
  - crop_path
  - mask_path
"""

import argparse
import csv
import math
import sys
from pathlib import Path

MASK_GEOMETRY_COLS = [
    "mask_area_px",
    "mask_fill_ratio",
    "bbox_w_px",
    "bbox_h_px",
    "bbox_aspect_ratio",
    "geometry_compactness",
]


def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return default if math.isnan(out) or math.isinf(out) else out
    except (TypeError, ValueError):
        return default


def _bootstrap_import_paths():
    here = Path(__file__).resolve()
    package_root = here.parents[3]
    repo_root = package_root.parent
    for path in (str(package_root), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


def load_extract_geometry_features():
    _bootstrap_import_paths()
    try:
        from scripts.stage2a.common import extract_geometry_features
    except ImportError:  # pragma: no cover
        from II_package.scripts.stage2a.common import extract_geometry_features
    return extract_geometry_features


def parse_args():
    p = argparse.ArgumentParser(description="Build Stage-2a inference CSV from shared instance samples.")
    p.add_argument("--shared_csv", type=Path, required=True, help="Path to shared_instance_samples.csv")
    p.add_argument("--out_csv", type=Path, required=True, help="Output CSV for Stage-2a inference")
    p.add_argument("--id_col", type=str, default="bldg_uid", help="Instance id column in shared CSV")
    p.add_argument("--crop_col", type=str, default="pre_crop", help="Image crop column in shared CSV")
    p.add_argument("--mask_col", type=str, default="mask_M", help="Mask column in shared CSV")
    p.add_argument("--tile_col", type=str, default="tile_id", help="Optional tile id column")
    p.add_argument("--event_col", type=str, default="event_id", help="Optional event id column")
    p.add_argument(
        "--extra_cols",
        type=str,
        default="sam3_confidence,hazard_type,classification_source,footprint_m2,log_footprint_m2,estimated_units,GEOID,tile_base",
        help="Optional comma-separated columns to carry through when present",
    )
    p.add_argument(
        "--add_mask_geometry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add mask-derived geometry columns used by refined Stage-2a checkpoints",
    )
    p.add_argument(
        "--derive_footprint_from_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Derive footprint_m2/log_footprint_m2 from mask area when no footprint columns are present",
    )
    p.add_argument(
        "--footprint_m2_per_mask_px",
        type=float,
        default=4.641392,
        help="Scale used when deriving footprint_m2 from mask_area_px; default is the Stage-2a train-manifest median ratio",
    )
    p.add_argument("--limit", type=int, default=0, help="Optional row limit for quick runs")
    p.add_argument(
        "--skip_missing_paths",
        action="store_true",
        help="Skip rows where crop/mask paths do not exist instead of raising an error",
    )
    p.add_argument("--log_every", type=int, default=200, help="Progress log interval")
    return p.parse_args()


def _must_have(columns, name):
    if name not in columns:
        raise KeyError(f"Missing required column '{name}'. Available columns: {sorted(columns)}")


def main():
    args = parse_args()
    if not args.shared_csv.exists():
        raise FileNotFoundError(f"Missing shared CSV: {args.shared_csv}")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    extract_geometry_features = load_extract_geometry_features() if (args.add_mask_geometry or args.derive_footprint_from_mask) else None

    with args.shared_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {args.shared_csv}")
        cols = set(reader.fieldnames)
        _must_have(cols, args.id_col)
        _must_have(cols, args.crop_col)
        _must_have(cols, args.mask_col)

        extra_cols = [x.strip() for x in args.extra_cols.split(",") if x.strip()]
        passthrough = []
        for c in [args.tile_col, args.event_col] + extra_cols:
            if c and c in cols and c not in passthrough:
                passthrough.append(c)

        mask_geometry_fields = list(MASK_GEOMETRY_COLS) if args.add_mask_geometry else []
        footprint_fields = ["footprint_m2", "log_footprint_m2", "footprint_m2_source"] if args.derive_footprint_from_mask else []
        out_fields = list(dict.fromkeys(["building_uid", "crop_path", "mask_path"] + passthrough + mask_geometry_fields + footprint_fields))
        rows_out = []
        n_in = 0
        n_skipped_missing = 0
        for row in reader:
            n_in += 1
            if args.limit > 0 and len(rows_out) >= args.limit:
                break

            crop_path = row.get(args.crop_col, "")
            mask_path = row.get(args.mask_col, "")
            if not crop_path or not mask_path:
                raise ValueError(f"Empty crop/mask path at input row {n_in}")

            crop_ok = Path(crop_path).exists()
            mask_ok = Path(mask_path).exists()
            if not (crop_ok and mask_ok):
                msg = (
                    f"Missing crop/mask file at row {n_in}: "
                    f"crop_exists={crop_ok}, mask_exists={mask_ok}, "
                    f"crop='{crop_path}', mask='{mask_path}'"
                )
                if args.skip_missing_paths:
                    n_skipped_missing += 1
                    if n_skipped_missing <= 3:
                        print("[warn]", msg)
                    continue
                raise FileNotFoundError(msg)

            out_row = {
                "building_uid": row.get(args.id_col, ""),
                "crop_path": crop_path,
                "mask_path": mask_path,
            }
            for c in passthrough:
                out_row[c] = row.get(c, "")
            geom = {}
            if args.add_mask_geometry:
                if extract_geometry_features is None:
                    raise RuntimeError("Mask geometry helper was not loaded")
                geom = extract_geometry_features(mask_path, row)
                for c in mask_geometry_fields:
                    out_row[c] = geom.get(c, "")
            if args.derive_footprint_from_mask:
                footprint_raw = out_row.get("footprint_m2") or row.get("footprint_m2") or ""
                log_footprint_raw = out_row.get("log_footprint_m2") or row.get("log_footprint_m2") or ""
                if footprint_raw:
                    footprint = safe_float(footprint_raw, 0.0)
                    out_row["footprint_m2"] = footprint
                    out_row["log_footprint_m2"] = safe_float(log_footprint_raw, math.log1p(max(0.0, footprint))) if log_footprint_raw else math.log1p(max(0.0, footprint))
                    out_row["footprint_m2_source"] = row.get("footprint_m2_source", "input")
                else:
                    if not geom:
                        if extract_geometry_features is None:
                            raise RuntimeError("Mask geometry helper was not loaded")
                        geom = extract_geometry_features(mask_path, row)
                    footprint = float(geom.get("mask_area_px", 0.0)) * float(args.footprint_m2_per_mask_px)
                    out_row["footprint_m2"] = footprint
                    out_row["log_footprint_m2"] = math.log1p(max(0.0, footprint))
                    out_row["footprint_m2_source"] = "mask_area_px_scaled"
            rows_out.append(out_row)

            if len(rows_out) % max(1, args.log_every) == 0:
                print(f"[build_stage2a_infer_csv] prepared={len(rows_out)}")

    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows_out)

    print("[done] input_rows=", n_in)
    print("[done] output_rows=", len(rows_out))
    print("[done] skipped_missing_paths=", n_skipped_missing)
    print("[done] wrote=", args.out_csv)


if __name__ == "__main__":
    main()
