#!/usr/bin/env python3
"""Draw the illustrative Stage 1 output for packaged LA cell 00507."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "instance-impact-stage1-illustration",
    }
)

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from shapely import wkt
from shapely.geometry import box


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CELL_ROOT = REPO_ROOT / "II_package/la_fire_results/demo_cell_00507"
IMAGE_PATH = CELL_ROOT / "cell_00507_pre.tif"
CSV_PATH = CELL_ROOT / "multidate_inputs/dates/20250110/shared_for_date.csv"
OUTPUT_STEM = HERE / "stage1_instance_illustration"

BOUNDARY = "#FFD400"
HALO = "#111111"
ZOOM = (1000, 1000, 560)


def load_inputs() -> tuple[np.ndarray, list[np.ndarray], int]:
    with rasterio.open(IMAGE_PATH) as src:
        image = np.moveaxis(src.read((1, 2, 3)), 0, -1)

    height, width = image.shape[:2]
    image_extent = box(0, 0, width, height)
    segments: list[np.ndarray] = []
    excluded = 0
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            geometry = wkt.loads(row["polygon_wkt_xy_pre"])
            geometry = geometry.intersection(image_extent)
            if geometry.is_empty:
                excluded += 1
                continue
            polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
            for polygon in polygons:
                segments.append(np.asarray(polygon.exterior.coords))

    assert image.ndim == 3 and image.shape[2] == 3
    assert segments, "No in-frame Stage 1 polygons found"
    return image, segments, excluded


def draw_boundaries(ax: plt.Axes, segments: list[np.ndarray], *, zoomed: bool = False) -> None:
    halo_width, boundary_width = ((2.0, 1.05) if zoomed else (1.15, 0.58))
    ax.add_collection(LineCollection(segments, colors=HALO, linewidths=halo_width, alpha=0.88))
    ax.add_collection(LineCollection(segments, colors=BOUNDARY, linewidths=boundary_width))


def label_panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.025,
        0.965,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#111111",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none", "alpha": 0.94},
        zorder=20,
    )


def draw() -> None:
    image, segments, excluded = load_inputs()
    height, width = image.shape[:2]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.58), facecolor="white")
    for ax in axes:
        ax.imshow(image, origin="upper")
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis("off")

    label_panel(axes[0], "a")
    draw_boundaries(axes[1], segments)
    label_panel(axes[1], "b")

    zoom_x, zoom_y, zoom_size = ZOOM
    axes[1].add_patch(
        Rectangle(
            (zoom_x, zoom_y),
            zoom_size,
            zoom_size,
            fill=False,
            edgecolor="white",
            linewidth=1.1,
            zorder=12,
        )
    )
    inset = inset_axes(axes[1], width="42%", height="42%", loc="upper right", borderpad=0.7)
    inset.imshow(image, origin="upper")
    draw_boundaries(inset, segments, zoomed=True)
    inset.set_xlim(zoom_x, zoom_x + zoom_size)
    inset.set_ylim(zoom_y + zoom_size, zoom_y)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("white")
        spine.set_linewidth(1.4)

    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995, wspace=0.018)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=400, facecolor="white", metadata={"Software": None})
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), facecolor="white", metadata={"Date": None})
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), facecolor="white", metadata={"CreationDate": None})
    plt.close(fig)

    print(f"Rendered {len(segments)} in-frame boundaries; excluded {excluded} out-of-frame geometry.")


if __name__ == "__main__":
    draw()
