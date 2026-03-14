from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shapely.geometry import mapping


def write_geojson(features: list[dict[str, Any]], out_path: Path, crs: Any | None) -> None:
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(f["geometry"]),
                "properties": f["properties"],
            }
            for f in features
        ],
    }
    if crs is not None:
        fc["crs"] = {"type": "name", "properties": {"name": str(crs)}}
    out_path.write_text(json.dumps(fc))


def write_geopackage(features: list[dict[str, Any]], out_path: Path, crs: Any | None) -> bool:
    try:
        import geopandas as gpd
    except Exception:
        return False

    rows = [
        {
            **f["properties"],
            "geometry": f["geometry"],
        }
        for f in features
    ]
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    gdf.to_file(out_path, driver="GPKG")
    return True
