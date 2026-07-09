from __future__ import annotations

import json
import contextlib
import io
import sys
import uuid
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union


def _shapes_suppress_known_gdal_warnings(bin_mask):
    """Run rasterio.features.shapes while filtering known noisy GDAL deprecation lines."""
    err_buf = io.StringIO()
    with contextlib.redirect_stderr(err_buf):
        out = list(shapes(bin_mask, mask=(bin_mask > 0)))
    err_text = err_buf.getvalue()
    if err_text:
        filtered = []
        for ln in err_text.splitlines():
            if "Memory' driver is deprecated" in ln and "Use 'MEM' onwards" in ln:
                continue
            filtered.append(ln)
        if filtered:
            print("\n".join(filtered), file=sys.stderr)
    return out


def vectorize_mask_to_wkt_json(mask_path: Path, score_path: Path, output_dir: Path, epsilon: float = 2.0) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(mask_path) as msrc:
        mask_arr = msrc.read(1)
    if score_path.exists():
        with rasterio.open(score_path) as ssrc:
            score_arr = ssrc.read(1)
    else:
        score_arr = None

    features = []
    labels = sorted(int(x) for x in np.unique(mask_arr) if int(x) > 0)
    for mask_id in labels:
        bin_mask = (mask_arr == mask_id).astype(np.uint8)
        geoms = [shape(g) for g, v in _shapes_suppress_known_gdal_warnings(bin_mask) if int(v) == 1]
        if not geoms:
            continue
        geom = unary_union(geoms)
        if geom.is_empty:
            continue
        # Keep epsilon behavior as lightweight simplification while preserving topology.
        if epsilon and float(epsilon) > 0:
            geom = geom.simplify(float(epsilon), preserve_topology=True)
        if geom.is_empty:
            continue
        prob = None
        if score_arr is not None:
            pix = score_arr[mask_arr == mask_id]
            if pix.size > 0:
                prob = float(pix.mean())
        features.append(
            {
                "properties": {
                    "feature_type": "building",
                    "uid": str(uuid.uuid4()),
                    "label": mask_id,
                    "prob": prob,
                },
                "wkt": str(geom.wkt),
            }
        )

    output_json = {
        "features": {"xy": features, "lng_lat": []},
        "metadata": {
            "original_width": int(mask_arr.shape[1]),
            "original_height": int(mask_arr.shape[0]),
            "width": int(mask_arr.shape[1]),
            "height": int(mask_arr.shape[0]),
            "img_name": mask_path.stem + ".png",
        },
    }

    out_path = output_dir / f"{mask_path.stem}_prediction.json"
    with open(out_path, "w") as f:
        json.dump(output_json, f, indent=2)
    return out_path
