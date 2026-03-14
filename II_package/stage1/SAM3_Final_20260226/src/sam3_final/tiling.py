from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
from rasterio.transform import Affine

from .utils import ensure_dir


@dataclass
class TileInfo:
    image_id: str
    tile_id: str
    tile_path: Path
    x: int
    y: int
    width: int
    height: int
    transform: Affine | None
    io_time_s: float


def generate_tiles(
    image_path: str | Path,
    out_dir: str | Path,
    tile_size: int | None,
    overlap: int = 0,
    transform: Affine | None = None,
) -> list[TileInfo]:
    image_path = Path(image_path)
    image_id = image_path.stem
    tiles_dir = ensure_dir(Path(out_dir) / "tiles" / image_id)

    import time

    with Image.open(image_path) as im:
        width, height = im.size
        if not tile_size:
            tile_path = tiles_dir / f"{image_id}.png"
            t0 = time.perf_counter()
            im.save(tile_path)
            t1 = time.perf_counter()
            return [
                TileInfo(
                    image_id=image_id,
                    tile_id=f"{image_id}_full",
                    tile_path=tile_path,
                    x=0,
                    y=0,
                    width=width,
                    height=height,
                    transform=transform,
                    io_time_s=t1 - t0,
                )
            ]

        stride = max(1, tile_size - overlap)
        tiles: list[TileInfo] = []
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                x2 = min(x + tile_size, width)
                y2 = min(y + tile_size, height)
                if x2 - x <= 0 or y2 - y <= 0:
                    continue
                t0 = time.perf_counter()
                tile = im.crop((x, y, x2, y2))
                tile_id = f"{image_id}_x{x}_y{y}_w{x2-x}_h{y2-y}"
                tile_path = tiles_dir / f"{tile_id}.png"
                tile.save(tile_path)
                t1 = time.perf_counter()

                tile_transform = None
                if transform is not None:
                    tile_transform = transform * Affine.translation(x, y)

                tiles.append(
                    TileInfo(
                        image_id=image_id,
                        tile_id=tile_id,
                        tile_path=tile_path,
                        x=x,
                        y=y,
                        width=x2 - x,
                        height=y2 - y,
                        transform=tile_transform,
                        io_time_s=t1 - t0,
                    )
                )
        return tiles
