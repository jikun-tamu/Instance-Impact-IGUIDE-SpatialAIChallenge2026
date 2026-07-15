#!/usr/bin/env python3
"""Recreate II-Conceptualization-v4.png as an editable, self-contained SVG."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "svg.hashsalt": "ii-conceptualization-v4",
    }
)

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath


WIDTH = 1627
HEIGHT = 967
INK = "#111111"
WHITE = "#FFFFFF"


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    radius: float = 12,
    linewidth: float = 2.3,
    zorder: int = 2,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=WHITE,
            edgecolor=INK,
            linewidth=linewidth,
            zorder=zorder,
        )
    )


def text(
    ax: plt.Axes,
    x: float,
    y: float,
    value: str,
    *,
    size: float = 17,
    weight: str = "normal",
    align: str = "center",
    zorder: int = 5,
) -> None:
    ax.text(
        x,
        y,
        value,
        ha=align,
        va="center",
        fontsize=size,
        fontweight=weight,
        color=INK,
        linespacing=1.22,
        zorder=zorder,
    )


def arrow(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    *,
    linewidth: float = 2.4,
    head_size: float = 15,
    zorder: int = 3,
) -> None:
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    ax.add_patch(
        FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=head_size,
            linewidth=linewidth,
            color=INK,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def stack(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    layers: int = 4,
    dx: float = 18,
    dy: float = 24,
) -> None:
    for layer in range(layers - 1, 0, -1):
        box(
            ax,
            x - layer * dx,
            y - layer * dy,
            width,
            height,
            radius=3,
            linewidth=2.2,
            zorder=2,
        )
    box(ax, x, y, width, height, radius=3, linewidth=2.4, zorder=4)


def add_crop(
    ax: plt.Axes,
    image,
    source: tuple[int, int, int, int],
    target: tuple[int, int, int, int],
) -> None:
    sx, sy, sw, sh = source
    tx, ty, tw, th = target
    crop = image[sy : sy + sh, sx : sx + sw]
    ax.imshow(crop, extent=(tx, tx + tw, ty + th, ty), interpolation="none", zorder=5)


def draw(source_png: Path, output_svg: Path) -> None:
    image = plt.imread(source_png)
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100, facecolor=WHITE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis("off")

    # Overall frame and headings.
    box(ax, 17, 15, 1593, 934, radius=38, linewidth=2.8, zorder=1)
    text(ax, WIDTH / 2, 49, "Instance Impact: Conceptualization", size=28, weight="bold")

    box(ax, 34, 84, 1555, 356, radius=20, linewidth=2.6, zorder=1)
    text(ax, WIDTH / 2, 106, "Instance Impact: Models", size=16, weight="bold")

    # Four independent model-stage panels.
    box(ax, 47, 121, 337, 304, radius=12, linewidth=2.2)
    box(ax, 396, 121, 427, 304, radius=12, linewidth=2.2)
    box(ax, 834, 121, 457, 304, radius=12, linewidth=2.2)
    box(ax, 1304, 121, 269, 304, radius=12, linewidth=2.2)

    # Stage 1.
    box(ax, 63, 156, 168, 96, radius=3, linewidth=2.1)
    text(ax, 147, 204, "Pre-Disaster\nImages", size=14)
    box(ax, 194, 269, 164, 98, radius=3, linewidth=2.1)
    text(ax, 276, 318, "Polygons", size=14)
    arrow(ax, [(147, 252), (147, 312), (194, 312)])
    text(ax, 56, 405, "Stage 1 - SAM 3 API", size=13, align="left")

    # Stage 2a.
    for x, y, w, h, label in (
        (415, 145, 183, 99, "Per-building\nPolygons"),
        (628, 145, 175, 99, "Per-building\nSubimages"),
        (415, 269, 183, 100, "Building\nCategory"),
        (626, 269, 177, 100, "Model Target:\nPopulation"),
    ):
        box(ax, x, y, w, h, radius=3, linewidth=2.1)
        text(ax, x + w / 2, y + h / 2, label, size=14)
    arrow(ax, [(598, 195), (628, 195)])
    arrow(ax, [(715, 244), (715, 269)])
    arrow(ax, [(598, 319), (626, 319)])
    arrow(ax, [(598, 195), (615, 195), (615, 319), (626, 319)])
    text(ax, 410, 405, "Stage 2a - Building Population Estimation", size=13, align="left")

    # Stage 2b.
    for x, y, w, h, label in (
        (854, 145, 187, 100, "Per-building\nPolygons"),
        (1075, 145, 185, 100, "Per-building\nDamage Stage"),
        (854, 269, 187, 100, "Per-building\nSubimages"),
        (1075, 269, 188, 100, "Model Target:\nDamage State"),
    ):
        box(ax, x, y, w, h, radius=3, linewidth=2.1)
        text(ax, x + w / 2, y + h / 2, label, size=14)
    arrow(ax, [(1041, 195), (1075, 195)])
    arrow(ax, [(948, 245), (948, 269)])
    arrow(ax, [(1168, 245), (1168, 269)])
    arrow(ax, [(1041, 319), (1075, 319)])
    text(ax, 846, 405, "Stage 2b - Building Damage State Prediction", size=13, align="left")

    # Stage 3.
    box(ax, 1336, 212, 204, 116, radius=3, linewidth=2.1)
    text(ax, 1438, 270, "Building-Level\nResults", size=14)
    text(ax, 1314, 405, "Stage 3 - Multi-Date Synthesis", size=12.5, align="left")

    # Inference panel.
    box(ax, 36, 455, 1552, 472, radius=24, linewidth=2.6, zorder=1)
    text(ax, WIDTH / 2, 480, "Instance Impact: Inference Time", size=16, weight="bold")

    # Embedded raster thumbnails cropped from the accepted PNG.
    add_crop(ax, image, (64, 520, 130, 131), (64, 520, 130, 131))
    add_crop(ax, image, (208, 520, 131, 131), (208, 520, 131, 131))
    add_crop(ax, image, (447, 789, 115, 121), (447, 789, 115, 121))
    add_crop(ax, image, (624, 789, 117, 121), (624, 789, 117, 121))

    # Paired imagery and Stage 1 output stack.
    box(ax, 63, 676, 214, 107, radius=3, linewidth=2.2)
    text(ax, 170, 729, "Paired Pre- & Post-\nDisaster Images", size=14)
    stack(ax, 519, 679, 213, 103, layers=4, dx=18, dy=30)
    text(ax, 625, 730, "Building Instance\nSubimage", size=14)
    arrow(ax, [(277, 729), (519, 729)])
    text(ax, 325, 711, "Stage 1", size=13, align="left")
    arrow(ax, [(562, 849), (624, 849)])

    # Parallel Stage 2 branches.
    stack(ax, 943, 580, 260, 94, layers=4, dx=18, dy=25)
    text(
        ax,
        1073,
        627,
        "Building-Type Probabilities +\nConstructed Exposure Proxy",
        size=12.2,
    )
    stack(ax, 960, 792, 240, 102, layers=4, dx=18, dy=25)
    text(ax, 1080, 843, "Disaster Impact\nEvaluation", size=14)

    arrow(ax, [(732, 716), (769, 716), (769, 627), (943, 627)])
    text(ax, 808, 609, "Stage 2 a", size=13, align="left")
    arrow(ax, [(732, 744), (769, 744), (769, 843), (960, 843)])
    text(ax, 808, 824, "Stage 2 b", size=13, align="left")

    # Stage 3 convergence and final result stack.
    ax.plot([1203, 1246, 1246], [627, 627, 843], color=INK, linewidth=2.2, zorder=3)
    ax.plot([1200, 1246], [843, 843], color=INK, linewidth=2.2, zorder=3)
    stack(ax, 1340, 678, 191, 107, layers=3, dx=18, dy=25)
    arrow(ax, [(1246, 728), (1340, 728)])
    text(ax, 1278, 708, "Stage 3", size=12.5)
    text(ax, 1435, 731, "Paired\nOccupancy Unit +\nDisaster Impact", size=13.2)

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg, format="svg", facecolor=WHITE, metadata={"Date": None})
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "II_package" / "II-Conceptualization-v4.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "II_package" / "II-Conceptualization-v4.svg",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    draw(args.source.resolve(), args.output.resolve())
    print(f"Wrote {args.output.resolve()}")
