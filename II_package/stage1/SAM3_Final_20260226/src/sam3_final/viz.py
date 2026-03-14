from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import shape


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path)


def load_mask(path: str | Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1)


def colorize_instance_mask(mask: np.ndarray, seed: int = 0) -> Image.Image:
    random.seed(seed)
    h, w = mask.shape
    out = Image.new("RGB", (w, h), (0, 0, 0))
    pixels = out.load()
    labels = np.unique(mask)
    colors = {}
    for lbl in labels:
        if lbl == 0:
            colors[int(lbl)] = (0, 0, 0)
        else:
            colors[int(lbl)] = (random.randint(30, 230), random.randint(30, 230), random.randint(30, 230))
    for y in range(h):
        for x in range(w):
            pixels[x, y] = colors[int(mask[y, x])]
    return out


def draw_polygons(
    image: Image.Image,
    polygons: Iterable,
    color: str = "yellow",
    width: int = 2,
) -> Image.Image:
    img = image.copy()
    dr = ImageDraw.Draw(img)
    for geom in polygons:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            coords = list(geom.exterior.coords)
            dr.line(coords + [coords[0]], fill=color, width=width)
        elif geom.geom_type == "MultiPolygon":
            for g in geom.geoms:
                coords = list(g.exterior.coords)
                dr.line(coords + [coords[0]], fill=color, width=width)
    return img


def geojson_to_geoms(geojson: dict) -> list:
    feats = geojson.get("features", [])
    return [shape(f["geometry"]) for f in feats if f.get("geometry")]


def parse_tile_meta(name: str):
    import re

    m = re.search(r"_x(\\d+)_y(\\d+)_w(\\d+)_h(\\d+)", name)
    if not m:
        return None
    return tuple(int(x) for x in m.groups())


def stitch_instance_masks(tile_paths: Iterable[Path], full_size: tuple[int, int]) -> np.ndarray:
    w, h = full_size
    canvas = np.zeros((h, w), dtype=np.uint32)
    for tp in tile_paths:
        meta = parse_tile_meta(tp.stem)
        if meta is None:
            continue
        x, y, tw, th = meta
        tile_mask = load_mask(tp)
        canvas[y : y + th, x : x + tw] = np.maximum(canvas[y : y + th, x : x + tw], tile_mask[:th, :tw])
    return canvas


def stitch_annotation_tiles(tile_paths: Iterable[Path], full_size: tuple[int, int]) -> Image.Image:
    w, h = full_size
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    for tp in tile_paths:
        meta = parse_tile_meta(tp.stem)
        if meta is None:
            continue
        x, y, tw, th = meta
        tile_img = Image.open(tp)
        if tile_img.size != (tw, th):
            tile_img = tile_img.crop((0, 0, min(tw, tile_img.size[0]), min(th, tile_img.size[1])))
        canvas.paste(tile_img, (x, y))
    return canvas
