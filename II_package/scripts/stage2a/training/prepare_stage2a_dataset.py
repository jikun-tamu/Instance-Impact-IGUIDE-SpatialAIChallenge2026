#!/usr/bin/env python3
"""Prepare a locked Stage-2a dataset manifest from the original label package."""

from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping

try:
    from scripts.stage2a.common import attach_geometry_columns, save_json
except ImportError:  # pragma: no cover - package import path
    from II_package.scripts.stage2a.common import attach_geometry_columns, save_json


REQUIRED_COLUMNS = [
    "building_uid",
    "tile_base",
    "crop_path",
    "mask_path",
    "type_class",
    "type_class_name",
    "estimated_population",
]

MANIFEST_COLUMNS = [
    "building_uid",
    "tile_base",
    "crop_path",
    "mask_path",
    "type_class",
    "type_class_name",
    "building_type",
    "is_residential",
    "estimated_units",
    "estimated_population",
    "exposure_target_policy",
    "centroid_lng",
    "centroid_lat",
    "GEOID",
    "footprint_m2",
    "mask_area_px",
    "mask_fill_ratio",
    "bbox_w_px",
    "bbox_h_px",
    "bbox_aspect_ratio",
    "geometry_compactness",
    "classification_source",
    "occupancy_rate",
    "ppu_single_family_detached",
    "ppu_mobile_home",
    "ppu_duplex",
    "ppu_triplex",
    "ppu_multi_family",
    "ppu_small_multi_family",
    "ppu_large_multi_family",
    "ppu_unknown",
]


def validate_target_policy(target_policy: str) -> Dict[str, object]:
    if target_policy != "exposure_proxy":
        raise ValueError("Stage 2a does not have direct per-building population ground truth.")
    return {
        "target_policy": target_policy,
        "target_name": "estimated_population",
        "methodology_label": "exposure population proxy",
        "allow_non_residential": True,
    }


def read_rows(path: Path) -> List[MutableMapping[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise KeyError(f"Missing required columns in {path}: {missing}")
        return [dict(row) for row in reader]


def extract_source_zip(source_zip: Path, extract_root: Path, overwrite: bool = False) -> Path:
    if not source_zip.exists():
        raise FileNotFoundError(f"Missing source zip: {source_zip}")
    label_path = extract_root / "ml_dataset" / "labels.csv"
    if label_path.exists() and not overwrite:
        return label_path
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_zip) as zf:
        zf.extractall(extract_root)
    if not label_path.exists():
        matches = list(extract_root.rglob("labels.csv"))
        if not matches:
            raise FileNotFoundError(f"No labels.csv found after extracting {source_zip}")
        label_path = matches[0]
    return label_path


def _normalize_path(value: object, dataset_root: Path) -> str:
    raw = str(value or "")
    if not raw:
        return ""
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    return str((dataset_root / p).resolve())


def normalize_stage2a_paths(rows: Iterable[MutableMapping[str, object]], dataset_root: Path) -> List[MutableMapping[str, object]]:
    out = []
    for row in rows:
        row = dict(row)
        row["crop_path"] = _normalize_path(row.get("crop_path"), dataset_root)
        row["mask_path"] = _normalize_path(row.get("mask_path"), dataset_root)
        out.append(row)
    return out


def write_manifest(rows: Iterable[Mapping[str, object]], out_csv: Path, target_policy: str) -> None:
    validate_target_policy(target_policy)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            r = {c: row.get(c, "") for c in MANIFEST_COLUMNS}
            r["exposure_target_policy"] = target_policy
            writer.writerow(r)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare a locked Stage-2a manifest.")
    p.add_argument("--source_zip", type=Path, default=Path("stage2a_old_training_data_codes/Harris_CBG_Building_Population.zip"))
    p.add_argument("--extract_root", type=Path, default=Path("outputs/stage2a/source_harris_cbg"))
    p.add_argument("--labels_csv", type=Path, default=None, help="Use an already extracted labels.csv instead of --source_zip")
    p.add_argument("--out_manifest", type=Path, required=True)
    p.add_argument("--out_policy_json", type=Path, default=None)
    p.add_argument("--target_policy", type=str, default="exposure_proxy")
    p.add_argument("--overwrite_extract", action="store_true")
    p.add_argument("--skip_geometry", action="store_true", help="Do not read masks to attach derived geometry columns")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    policy = validate_target_policy(args.target_policy)
    label_path = args.labels_csv if args.labels_csv else extract_source_zip(args.source_zip, args.extract_root, args.overwrite_extract)
    dataset_root = label_path.parent.parent
    rows = normalize_stage2a_paths(read_rows(label_path), dataset_root)
    if not args.skip_geometry:
        rows = attach_geometry_columns(rows)
    write_manifest(rows, args.out_manifest, args.target_policy)
    if args.out_policy_json:
        save_json(args.out_policy_json, policy)
    print("[done] source_labels=", label_path)
    print("[done] rows=", len(rows))
    print("[done] wrote=", args.out_manifest)


if __name__ == "__main__":
    main()
