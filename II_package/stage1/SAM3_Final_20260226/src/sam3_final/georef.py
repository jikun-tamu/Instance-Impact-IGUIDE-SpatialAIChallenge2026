from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from rasterio.transform import Affine, from_bounds


WORLD_EXTS = [".wld", ".pgw", ".tfw", ".jgw"]


@dataclass
class GeoRef:
    width: int
    height: int
    crs: Any | None
    transform: Affine | None
    source: str


def _parse_world_file(path: Path) -> Affine:
    vals = [float(x.strip()) for x in path.read_text().strip().splitlines() if x.strip()]
    if len(vals) < 6:
        raise ValueError(f"World file {path} has {len(vals)} values; expected 6")
    a, d, b, e, c, f = vals[:6]
    return Affine(a, b, c, d, e, f)


def _find_world_file(image_path: Path) -> Path | None:
    for ext in WORLD_EXTS:
        cand = image_path.with_suffix(ext)
        if cand.exists():
            return cand
    return None


def _find_sibling_geotiff(image_path: Path) -> Path | None:
    for ext in [".tif", ".tiff", ".TIF", ".TIFF"]:
        cand = image_path.with_suffix(ext)
        if cand.exists():
            return cand
    return None


def _load_metadata_index(metadata_path: Path) -> dict[str, dict[str, Any]]:
    data = metadata_path.read_text()
    if metadata_path.suffix.lower() == ".json":
        items = json.loads(data)
        if isinstance(items, dict):
            items = items.get("items", [])
        out = {}
        for item in items:
            key = str(item.get("image_id") or item.get("id") or item.get("name"))
            if key and key != "None":
                out[key] = item
        return out
    if metadata_path.suffix.lower() == ".csv":
        import csv

        out = {}
        rows = list(csv.DictReader(data.splitlines()))
        for row in rows:
            key = row.get("image_id") or row.get("id") or row.get("name")
            if key:
                out[str(key)] = row
        return out
    raise ValueError(f"Unsupported metadata file type: {metadata_path.suffix}")


def _georef_from_metadata(item: dict[str, Any], width: int, height: int) -> GeoRef | None:
    crs = item.get("crs") or item.get("epsg")
    transform = None
    if item.get("transform"):
        vals = item["transform"]
        if isinstance(vals, str):
            vals = [float(v) for v in vals.replace("|", " ").replace(",", " ").split()]
        if len(vals) >= 6:
            transform = Affine(*vals[:6])
    elif item.get("bounds"):
        bounds = item["bounds"]
        if isinstance(bounds, str):
            bounds = [float(v) for v in bounds.replace(",", " ").split()]
        if len(bounds) >= 4:
            transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
    if crs or transform:
        return GeoRef(width=width, height=height, crs=crs, transform=transform, source="metadata")
    return None


def find_georef(image_path: str | Path, metadata_path: str | Path | None = None) -> GeoRef:
    image_path = Path(image_path)
    with Image.open(image_path) as im:
        width, height = im.size

    # 1) Sidecar world file
    world = _find_world_file(image_path)
    if world:
        return GeoRef(width=width, height=height, crs=None, transform=_parse_world_file(world), source=f"world:{world.name}")

    # 2) Sibling GeoTIFF with same base name
    sibling = _find_sibling_geotiff(image_path)
    if sibling:
        import rasterio

        with rasterio.open(sibling) as ds:
            return GeoRef(width=width, height=height, crs=ds.crs, transform=ds.transform, source=f"geotiff:{sibling.name}")

    # 3) User-provided metadata table
    if metadata_path:
        meta = _load_metadata_index(Path(metadata_path))
        key = image_path.stem
        if key in meta:
            ref = _georef_from_metadata(meta[key], width=width, height=height)
            if ref:
                return ref

    return GeoRef(width=width, height=height, crs=None, transform=None, source="none")
