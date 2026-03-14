#!/usr/bin/env python3
"""Build a Stage-2a inference CSV from shared instance artifacts.

Input contract:
- shared_instance_samples.csv produced by scripts/generate_shared_instance_subimages.py

Output contract:
- CSV with at least:
  - building_uid
  - crop_path
  - mask_path
"""

import argparse
import csv
from pathlib import Path


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
        default="sam3_confidence,hazard_type",
        help="Optional comma-separated columns to carry through when present",
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
            if c and c in cols:
                passthrough.append(c)

        out_fields = ["building_uid", "crop_path", "mask_path"] + passthrough
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
