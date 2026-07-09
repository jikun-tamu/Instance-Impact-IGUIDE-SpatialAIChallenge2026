from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import shape
from shapely.ops import unary_union

from .regularize import regularize_geometry


@dataclass
class PolygonizeConfig:
    regularize_method: str = "none"
    epsilon: float = 2.0
    use_geoai: bool = False


def _label_from_geom(mask_arr: np.ndarray, geom, transform: Affine) -> int:
    # Sample centroid label, fallback to majority within bounds
    try:
        cx, cy = geom.centroid.coords[0]
        row, col = ~transform * (cx, cy)
        row_i = int(round(row))
        col_i = int(round(col))
        if 0 <= row_i < mask_arr.shape[0] and 0 <= col_i < mask_arr.shape[1]:
            lbl = int(mask_arr[row_i, col_i])
            if lbl != 0:
                return lbl
    except Exception:
        pass

    try:
        minx, miny, maxx, maxy = geom.bounds
        r0, c0 = ~transform * (minx, maxy)
        r1, c1 = ~transform * (maxx, miny)
        r0 = max(0, int(np.floor(r0)))
        r1 = min(mask_arr.shape[0], int(np.ceil(r1)))
        c0 = max(0, int(np.floor(c0)))
        c1 = min(mask_arr.shape[1], int(np.ceil(c1)))
        window = mask_arr[r0:r1, c0:c1]
        if window.size > 0:
            counts = np.bincount(window.ravel())
            if len(counts) > 1:
                return int(np.argmax(counts[1:]) + 1)
    except Exception:
        pass
    return 0


def _mean_score_for_label(score_arr: np.ndarray | None, mask_arr: np.ndarray, label: int) -> float | None:
    if score_arr is None or label == 0:
        return None
    mask = mask_arr == label
    if not mask.any():
        return None
    return float(score_arr[mask].mean())


def polygonize_mask_rasterio(
    mask_path: Path,
    score_path: Path | None,
    transform: Affine | None,
    cfg: PolygonizeConfig,
) -> list[dict[str, Any]]:
    with rasterio.open(mask_path) as src:
        mask_arr = src.read(1)
        if transform is None:
            transform = src.transform if src.transform is not None else Affine.identity()

    score_arr = None
    if score_path and score_path.exists():
        with rasterio.open(score_path) as ssrc:
            score_arr = ssrc.read(1)

    # Collect geometries per label
    label_geoms: dict[int, list] = {}
    for geom_mapping, value in shapes(mask_arr, mask=mask_arr > 0, transform=transform):
        lbl = int(value)
        if lbl == 0:
            continue
        geom = shape(geom_mapping)
        if geom.is_empty:
            continue
        label_geoms.setdefault(lbl, []).append(geom)

    features: list[dict[str, Any]] = []
    for lbl, geoms in label_geoms.items():
        geom = unary_union(geoms)
        if geom.is_empty:
            continue
        geom = regularize_geometry(geom, cfg.regularize_method, cfg.epsilon)
        if geom.is_empty:
            continue
        prob = _mean_score_for_label(score_arr, mask_arr, lbl)
        features.append(
            {
                "geometry": geom,
                "properties": {
                    "uid": str(uuid.uuid4()),
                    "instance_id": int(lbl),
                    "confidence": prob,
                },
            }
        )
    return features


def polygonize_mask_geoai(
    mask_path: Path,
    score_path: Path | None,
    cfg: PolygonizeConfig,
    output_transform: Affine | None,
) -> list[dict[str, Any]]:
    import geoai

    tmp_geojson = mask_path.with_suffix(".tmp.geojson")
    gdf = geoai.orthogonalize(str(mask_path), str(tmp_geojson), epsilon=cfg.epsilon)
    if gdf is None or len(gdf) == 0:
        if tmp_geojson.exists():
            tmp_geojson.unlink()
        return []

    with rasterio.open(mask_path) as src:
        mask_arr = src.read(1)
        transform = src.transform if src.transform is not None else Affine.identity()

    score_arr = None
    if score_path and score_path.exists():
        with rasterio.open(score_path) as ssrc:
            score_arr = ssrc.read(1)

    features: list[dict[str, Any]] = []
    from shapely.affinity import affine_transform

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        lbl = _label_from_geom(mask_arr, geom, transform)
        prob = _mean_score_for_label(score_arr, mask_arr, lbl)
        if output_transform is not None:
            a, b, c, d, e, f = output_transform.a, output_transform.b, output_transform.c, output_transform.d, output_transform.e, output_transform.f
            geom = affine_transform(geom, [a, b, d, e, c, f])
        features.append(
            {
                "geometry": geom,
                "properties": {
                    "uid": str(uuid.uuid4()),
                    "instance_id": int(lbl),
                    "confidence": prob,
                },
            }
        )

    if tmp_geojson.exists():
        tmp_geojson.unlink()
    return features


def polygonize_mask(
    mask_path: Path,
    score_path: Path | None,
    transform: Affine | None,
    cfg: PolygonizeConfig,
) -> list[dict[str, Any]]:
    if cfg.use_geoai:
        return polygonize_mask_geoai(mask_path, score_path, cfg, transform)
    return polygonize_mask_rasterio(mask_path, score_path, transform, cfg)
