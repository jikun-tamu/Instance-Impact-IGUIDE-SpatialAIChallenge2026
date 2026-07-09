from __future__ import annotations

from shapely.geometry.base import BaseGeometry


def regularize_geometry(geom: BaseGeometry, method: str, epsilon: float) -> BaseGeometry:
    method = (method or "none").lower()
    if method == "none":
        return geom
    if method == "simplify":
        return geom.simplify(epsilon, preserve_topology=True)
    if method == "min_rot_rect":
        return geom.minimum_rotated_rectangle
    return geom
