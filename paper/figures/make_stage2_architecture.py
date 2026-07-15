#!/usr/bin/env python3
"""Draw the publication-facing Stage 2a and Stage 2b architecture figure."""

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
        "svg.hashsalt": "instance-impact-stage2-architecture",
    }
)

import matplotlib.pyplot as plt
from PIL import Image

from make_ii_conceptualization_v4_svg import arrow, box, text


WIDTH = 1600
HEIGHT = 880
INK = "#111111"
MUTED = "#5E5E5E"
WHITE = "#FFFFFF"


def note(ax: plt.Axes, x: float, y: float, value: str, *, align: str = "center", size: float = 11.5) -> None:
    ax.text(
        x,
        y,
        value,
        ha=align,
        va="center",
        fontsize=size,
        color=MUTED,
        linespacing=1.18,
        zorder=6,
    )


def node(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    detail: str = "",
) -> None:
    box(ax, x, y, width, height, radius=8, linewidth=2.1)
    title_y = y + height / 2 - (10 if detail else 0)
    text(ax, x + width / 2, title_y, title, size=14.5, weight="bold")
    if detail:
        note(ax, x + width / 2, y + height - 18, detail, size=10.5)


def draw(output_svg: Path, output_png: Path) -> None:
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100, facecolor=WHITE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis("off")

    # Panel a: Stage 2a.
    box(ax, 25, 25, 1550, 400, radius=22, linewidth=2.5, zorder=1)
    text(ax, 48, 58, "a", size=20, weight="bold", align="left")
    text(
        ax,
        88,
        58,
        "Stage 2a: building type and constructed exposure",
        size=20,
        weight="bold",
        align="left",
    )
    note(ax, 705, 96, "five independently fitted type classifiers", size=11.5)

    node(ax, 55, 115, 220, 82, "Pre-event RGB +\nfootprint mask", "4 × 224 × 224")
    node(ax, 55, 275, 220, 82, "Geometry vector", "log area · fill\naspect · compactness")
    node(ax, 330, 115, 230, 82, "ConvNeXt-Tiny +\nmask-aware pooling", "768 → 512")
    node(ax, 330, 275, 230, 82, "Geometry projection", "4 → 64")
    node(ax, 620, 195, 205, 88, "Feature fusion", "512 + 64")
    node(ax, 875, 195, 210, 88, "Three-class\ntype head", "576 → 128 → 3")
    node(ax, 1135, 115, 190, 82, "Probability mean", "five type vectors")
    node(ax, 1135, 275, 190, 82, "Ridge baseline +\nresidual HGB", "log area + soft type")
    node(ax, 1380, 115, 165, 82, "Building-type\nprobabilities")
    node(ax, 1380, 275, 165, 82, "Constructed\nexposure proxy")

    arrow(ax, [(275, 156), (330, 156)])
    arrow(ax, [(275, 316), (330, 316)])
    arrow(ax, [(560, 156), (590, 156), (590, 223), (620, 223)])
    arrow(ax, [(560, 316), (590, 316), (590, 255), (620, 255)])
    arrow(ax, [(825, 239), (875, 239)])
    arrow(ax, [(1085, 239), (1110, 239), (1110, 156), (1135, 156)])
    arrow(ax, [(1325, 156), (1380, 156)])
    arrow(ax, [(1230, 197), (1230, 275)])
    arrow(ax, [(275, 345), (300, 345), (300, 390), (1105, 390), (1105, 316), (1135, 316)])
    note(ax, 720, 373, "log footprint area", size=10.5)
    arrow(ax, [(1325, 316), (1380, 316)])

    # Panel b: Stage 2b.
    box(ax, 25, 455, 1550, 400, radius=22, linewidth=2.5, zorder=1)
    text(ax, 48, 488, "b", size=20, weight="bold", align="left")
    text(
        ax,
        88,
        488,
        "Stage 2b: ordinal damage from paired observations",
        size=20,
        weight="bold",
        align="left",
    )

    node(ax, 55, 555, 205, 80, "Pre-event RGB", "3 × 256 × 256")
    node(ax, 55, 690, 205, 80, "Post-event RGB", "3 × 256 × 256")
    node(ax, 325, 620, 220, 88, "Shared\nConvNeXt-Tiny", "C = 768")
    node(ax, 610, 620, 225, 88, "Footprint + context\nmasked pooling", "regions M and R")
    node(ax, 900, 620, 230, 88, "Pre/post + signed/\nabsolute change", "8C = 6144")
    node(ax, 1195, 620, 205, 88, "Cumulative\nordinal head", "6144 → 512 → 3")
    node(ax, 1450, 620, 100, 88, "Four\ndamage\nstates")

    arrow(ax, [(260, 595), (292, 595), (292, 642), (325, 642)])
    arrow(ax, [(260, 730), (292, 730), (292, 686), (325, 686)])
    arrow(ax, [(545, 664), (610, 664)])
    arrow(ax, [(835, 664), (900, 664)])
    arrow(ax, [(1130, 664), (1195, 664)])
    arrow(ax, [(1400, 664), (1450, 664)])

    box(ax, 850, 748, 590, 76, radius=8, linewidth=1.8)
    text(ax, 1145, 769, "Inference evidence", size=12.5, weight="bold")
    note(
        ax,
        1145,
        802,
        "weighted cumulative logits determine class; calibrated probabilities supply\nconfidence and expected severity",
        size=10.8,
    )
    arrow(ax, [(1298, 708), (1298, 748)], linewidth=1.8, head_size=12)

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg, format="svg", facecolor=WHITE, metadata={"Date": None})
    fig.savefig(output_png, format="png", dpi=200, facecolor=WHITE, metadata={"Software": None})
    plt.close(fig)
    with Image.open(output_png) as image:
        image.convert("RGB").save(output_png, optimize=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for stage2_model_architecture.svg and .png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    draw(
        args.out_dir / "stage2_model_architecture.svg",
        args.out_dir / "stage2_model_architecture.png",
    )


if __name__ == "__main__":
    main()
