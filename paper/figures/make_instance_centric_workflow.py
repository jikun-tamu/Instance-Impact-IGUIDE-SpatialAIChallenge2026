#!/usr/bin/env python3
"""Create faithful vector redraws of the two Instance Impact workflows.

Figure contract
---------------
Core conclusion: A persistent building instance links four explicitly named
stages to reviewable decision products, while the Stage 2a training targets
retain an auditable lineage from spatial and demographic source data.
Archetype: two complementary, schematic-led, full-width diagrams.
Hero evidence: the supplied two-band framework conceptualization.
Validation/provenance evidence: the supplied four-row population-proxy rule.
Statistics/source data: none; this is vector line art based on documented
pipeline contracts rather than a quantitative result panel.
Image integrity: no raster source imagery is embedded or altered.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath


INK = "#25313C"
MUTED = "#66717D"
LINE = "#52606D"
NEUTRAL = "#F3F5F7"
BLUE = "#3B6FB6"
BLUE_LIGHT = "#E8F0FA"
TEAL = "#2A8F83"
TEAL_LIGHT = "#DDF1EE"
PURPLE = "#7656A8"
PURPLE_LIGHT = "#EEE7F7"
ORANGE = "#D88332"
ORANGE_LIGHT = "#FBEBD9"
AMBER = "#B66A18"
AMBER_LIGHT = "#FFF1D6"
WHITE = "#FFFFFF"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    face: str,
    edge: str = LINE,
    linewidth: float = 0.9,
    radius: float = 0.018,
    linestyle: str = "-",
    zorder: int = 3,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def labeled_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    body: Sequence[str] = (),
    face: str = NEUTRAL,
    edge: str = LINE,
    title_color: str = INK,
    title_size: float = 7.0,
    body_size: float = 7.0,
    body_color: str = INK,
    linewidth: float = 0.9,
    linestyle: str = "-",
    align: str = "center",
) -> None:
    rounded_box(
        ax,
        x,
        y,
        w,
        h,
        face=face,
        edge=edge,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    tx = x + w / 2 if align == "center" else x + 0.014
    ha = "center" if align == "center" else "left"
    if body:
        ax.text(
            tx,
            y + h - 0.030,
            title,
            ha=ha,
            va="top",
            fontsize=title_size,
            fontweight="bold",
            color=title_color,
            linespacing=1.05,
            zorder=5,
        )
        ax.text(
            tx,
            y + h - 0.078,
            "\n".join(body),
            ha=ha,
            va="top",
            fontsize=body_size,
            color=body_color,
            linespacing=1.24,
            zorder=5,
        )
    else:
        ax.text(
            tx,
            y + h / 2,
            title,
            ha=ha,
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=title_color,
            linespacing=1.12,
            zorder=5,
        )


def route_arrow(
    ax: plt.Axes,
    points: Iterable[tuple[float, float]],
    *,
    color: str = LINE,
    linewidth: float = 1.0,
    linestyle: str = "-",
    mutation_scale: float = 7.5,
    zorder: int = 2,
) -> None:
    vertices = list(points)
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices) - 1)
    path = MplPath(vertices, codes)
    arrow = FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def panel_heading(ax: plt.Axes, letter: str, title: str) -> None:
    title_x = 0.0
    if letter:
        ax.text(
            0.0,
            0.985,
            letter,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color=INK,
        )
        title_x = 0.035
    ax.text(
        title_x,
        0.985,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=INK,
    )


def draw_asset_icon(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    kind: str,
    label: str,
) -> None:
    if kind in {"pre", "post"}:
        face = BLUE_LIGHT if kind == "pre" else ORANGE_LIGHT
        ax.add_patch(
            Rectangle(
                (x, y), w, h, facecolor=face, edgecolor=LINE, linewidth=0.55, zorder=5
            )
        )
        ax.plot(
            [x + 0.004, x + w * 0.44, x + w * 0.66, x + w - 0.004],
            [y + h * 0.28, y + h * 0.64, y + h * 0.43, y + h * 0.72],
            color=TEAL if kind == "pre" else ORANGE,
            linewidth=0.75,
            zorder=6,
        )
    else:
        ax.add_patch(
            Rectangle(
                (x, y), w, h, facecolor=INK, edgecolor=LINE, linewidth=0.55, zorder=5
            )
        )
        footprint = [
            (x + w * 0.23, y + h * 0.20),
            (x + w * 0.70, y + h * 0.16),
            (x + w * 0.79, y + h * 0.62),
            (x + w * 0.55, y + h * 0.82),
            (x + w * 0.18, y + h * 0.66),
        ]
        if kind == "mask":
            ax.add_patch(
                Polygon(footprint, closed=True, facecolor=WHITE, edgecolor=WHITE, zorder=6)
            )
        else:
            ax.add_patch(
                Polygon(
                    footprint,
                    closed=True,
                    facecolor="none",
                    edgecolor=WHITE,
                    linewidth=2.0,
                    zorder=6,
                )
            )
    ax.text(
        x + w / 2,
        y - 0.012,
        label,
        ha="center",
        va="top",
        fontsize=7.0,
        color=MUTED,
        zorder=6,
    )


def stacked_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    body: Sequence[str] = (),
    face: str = NEUTRAL,
    edge: str = LINE,
    title_color: str = INK,
    layers: int = 3,
    title_size: float = 8.0,
    body_size: float = 7.7,
) -> None:
    """Draw a compact stack, echoing the repeated-instance cards in the source."""
    for layer in range(layers - 1, 0, -1):
        offset = layer * 0.009
        rounded_box(
            ax,
            x - offset,
            y + offset,
            w,
            h,
            face=WHITE,
            edge=edge,
            linewidth=0.65,
            radius=0.010,
            zorder=2 + layer,
        )
    labeled_box(
        ax,
        x,
        y,
        w,
        h,
        title=title,
        body=body,
        face=face,
        edge=edge,
        title_color=title_color,
        title_size=title_size,
        body_size=body_size,
        linewidth=0.85,
    )


def stage_card(
    ax: plt.Axes,
    x: float,
    *,
    stage: str,
    name: Sequence[str],
    face: str,
    accent: str,
) -> None:
    """Draw one named stage in the upper framework band."""
    y, w, h = 0.585, 0.205, 0.235
    rounded_box(ax, x, y, w, h, face=face, edge=accent, linewidth=0.9, radius=0.014)
    ax.text(
        x + w / 2,
        y + h - 0.026,
        stage,
        ha="center",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=accent,
        zorder=5,
    )
    ax.text(
        x + w / 2,
        y + 0.105,
        "\n".join(name),
        ha="center",
        va="center",
        fontsize=7.7,
        fontweight="bold",
        color=INK,
        linespacing=1.08,
        zorder=5,
    )


def draw_panel_a(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rounded_box(ax, 0.008, 0.018, 0.984, 0.964, face=WHITE, edge=INK, linewidth=1.15, radius=0.026, zorder=1)
    ax.text(
        0.5,
        0.947,
        "Instance Impact: Instance-Centric GeoAI Framework",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
        color=INK,
        zorder=5,
    )

    rounded_box(ax, 0.026, 0.545, 0.948, 0.348, face="#FBFCFD", edge=LINE, linewidth=0.9, radius=0.019, zorder=1)
    ax.text(0.5, 0.862, "Framework stages", ha="center", va="center", fontsize=8.0, fontweight="bold", color=INK, zorder=5)
    stage_card(
        ax,
        0.046,
        stage="Stage 1",
        name=("Building-Instance", "Delineation and Asset", "Generation"),
        face=BLUE_LIGHT,
        accent=BLUE,
    )
    stage_card(
        ax,
        0.282,
        stage="Stage 2a",
        name=("Building-Type and", "Exposure-Proxy", "Estimation"),
        face=PURPLE_LIGHT,
        accent=PURPLE,
    )
    stage_card(
        ax,
        0.518,
        stage="Stage 2b",
        name=("Ordinal Building-Damage", "Estimation"),
        face=ORANGE_LIGHT,
        accent=ORANGE,
    )
    stage_card(
        ax,
        0.754,
        stage="Stage 3",
        name=("Coverage-Aware Multi-Date", "Synthesis"),
        face=TEAL_LIGHT,
        accent=TEAL,
    )

    rounded_box(ax, 0.026, 0.058, 0.948, 0.447, face="#FBFCFD", edge=LINE, linewidth=0.9, radius=0.019, zorder=1)
    ax.text(0.5, 0.478, "End-to-end inference", ha="center", va="center", fontsize=8.0, fontweight="bold", color=INK, zorder=5)

    labeled_box(ax, 0.046, 0.302, 0.105, 0.087, title="Pre-event\nimagery", face=BLUE_LIGHT, edge=BLUE, title_color=BLUE, title_size=7.8)
    labeled_box(ax, 0.046, 0.158, 0.105, 0.087, title="Post-event\nimagery", face=ORANGE_LIGHT, edge=ORANGE, title_color=AMBER, title_size=7.8)
    labeled_box(
        ax,
        0.188,
        0.226,
        0.135,
        0.150,
        title="Stage 1",
        body=("instances + assets",),
        face=BLUE_LIGHT,
        edge=BLUE,
        title_color=BLUE,
        title_size=8.3,
        body_size=7.7,
    )
    stacked_box(
        ax,
        0.382,
        0.205,
        0.165,
        0.174,
        title="Shared building instance",
        body=("UID • polygon", "aligned crops • masks"),
        face=TEAL_LIGHT,
        edge=TEAL,
        title_color=TEAL,
    )
    labeled_box(ax, 0.575, 0.286, 0.195, 0.132, title="Stage 2a", body=("pre crop • mask", "geometry → type + proxy"), face=PURPLE_LIGHT, edge=PURPLE, title_color=PURPLE, title_size=8.3, body_size=7.7)
    labeled_box(ax, 0.575, 0.132, 0.195, 0.132, title="Stage 2b", body=("paired pre/post • masks", "damage + uncertainty"), face=ORANGE_LIGHT, edge=ORANGE, title_color=AMBER, title_size=8.3, body_size=7.7)
    labeled_box(ax, 0.795, 0.232, 0.165, 0.145, title="Stage 3", body=("coverage-aware", "multi-date synthesis"), face=TEAL_LIGHT, edge=TEAL, title_color=TEAL, title_size=8.3, body_size=7.7)
    stacked_box(ax, 0.795, 0.073, 0.165, 0.125, title="Decision products", body=("instance table • maps", "review queue"), face=NEUTRAL, edge=INK, title_color=INK, layers=2, title_size=8.0, body_size=7.7)

    route_arrow(ax, [(0.151, 0.346), (0.180, 0.346)])
    route_arrow(ax, [(0.323, 0.301), (0.373, 0.301)])
    route_arrow(ax, [(0.151, 0.202), (0.352, 0.202), (0.352, 0.244), (0.373, 0.244)])
    route_arrow(ax, [(0.547, 0.315), (0.562, 0.352), (0.566, 0.352)])
    route_arrow(ax, [(0.547, 0.255), (0.562, 0.198), (0.566, 0.198)])
    route_arrow(ax, [(0.770, 0.352), (0.780, 0.352), (0.786, 0.329)])
    route_arrow(ax, [(0.770, 0.198), (0.780, 0.198), (0.786, 0.272)])
    route_arrow(ax, [(0.878, 0.232), (0.878, 0.207)])


def draw_panel_b(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rounded_box(ax, 0.008, 0.020, 0.984, 0.958, face=WHITE, edge=INK, linewidth=1.15, radius=0.026, zorder=1)
    ax.text(0.5, 0.944, "Stage 2a: Building-Type and Exposure-Proxy Training Targets", ha="center", va="center", fontsize=10.2, fontweight="bold", color=INK, zorder=5)

    source_y = [0.704, 0.508, 0.312, 0.116]
    sources = [
        ("Harris parcels", ("land use + units",), PURPLE_LIGHT, PURPLE),
        ("xBD Harvey", ("image + polygon",), BLUE_LIGHT, BLUE),
        ("CBG boundaries", ("GEOID",), TEAL_LIGHT, TEAL),
        ("ACS 2018–2022", ("PPU + occupancy",), AMBER_LIGHT, AMBER),
    ]
    derived = [
        ("Parcel evidence", ("type + units",), PURPLE_LIGHT, PURPLE),
        ("Building geometry", (), BLUE_LIGHT, BLUE),
        ("Assign CBG GEOID", (), TEAL_LIGHT, TEAL),
        ("CBG demographics", ("PPU + occupancy",), AMBER_LIGHT, AMBER),
    ]
    for sy, source, product in zip(source_y, sources, derived):
        labeled_box(ax, 0.032, sy, 0.176, 0.132, title=source[0], body=source[1], face=source[2], edge=source[3], title_color=source[3], title_size=7.8, body_size=7.7)
        labeled_box(ax, 0.270, sy, 0.190, 0.132, title=product[0], body=product[1], face=product[2], edge=product[3], title_color=product[3], title_size=7.8, body_size=7.7)
        route_arrow(ax, [(0.208, sy + 0.066), (0.261, sy + 0.066)])

    labeled_box(ax, 0.548, 0.704, 0.194, 0.132, title="Building type + unit rules", face=PURPLE_LIGHT, edge=PURPLE, title_color=PURPLE, title_size=7.8)
    labeled_box(ax, 0.548, 0.406, 0.194, 0.156, title="Geometry + CBG GEOID", body=("one record per building",), face=TEAL_LIGHT, edge=TEAL, title_color=TEAL, title_size=7.8, body_size=7.7)
    labeled_box(ax, 0.548, 0.116, 0.194, 0.132, title="CBG exposure parameters", body=("PPU + occupancy",), face=AMBER_LIGHT, edge=AMBER, title_color=AMBER, title_size=7.8, body_size=7.7)

    labeled_box(ax, 0.802, 0.718, 0.160, 0.112, title="Five type categories", face=PURPLE_LIGHT, edge=PURPLE, title_color=PURPLE, title_size=7.7)
    labeled_box(ax, 0.807, 0.570, 0.150, 0.100, title="Exclude", body=("institutional + other",), face=AMBER_LIGHT, edge=AMBER, title_color=AMBER, title_size=7.8, body_size=7.7)
    labeled_box(ax, 0.792, 0.216, 0.180, 0.240, title="Three-class fitting set", body=("type label", "building-level", "exposure proxy"), face=PURPLE_LIGHT, edge=PURPLE, title_color=PURPLE, title_size=7.8, body_size=7.7)

    route_arrow(ax, [(0.460, 0.770), (0.539, 0.770)])
    route_arrow(ax, [(0.460, 0.574), (0.506, 0.574), (0.506, 0.736), (0.539, 0.736)])
    route_arrow(ax, [(0.460, 0.574), (0.510, 0.574), (0.510, 0.505), (0.539, 0.505)])
    route_arrow(ax, [(0.460, 0.378), (0.510, 0.378), (0.510, 0.463), (0.539, 0.463)])
    route_arrow(ax, [(0.460, 0.182), (0.539, 0.182)])

    route_arrow(ax, [(0.742, 0.770), (0.793, 0.774)])
    route_arrow(ax, [(0.882, 0.718), (0.882, 0.679)])
    route_arrow(ax, [(0.882, 0.570), (0.882, 0.465)])
    route_arrow(ax, [(0.742, 0.727), (0.768, 0.727), (0.768, 0.397), (0.783, 0.397)])
    route_arrow(ax, [(0.742, 0.473), (0.770, 0.473), (0.770, 0.351), (0.783, 0.351)])
    route_arrow(ax, [(0.742, 0.182), (0.770, 0.182), (0.770, 0.298), (0.783, 0.298)])


def build_overview_figure() -> plt.Figure:
    fig = plt.figure(figsize=(7.2, 4.35), facecolor=WHITE)
    draw_panel_a(fig.add_subplot(1, 1, 1))
    fig.subplots_adjust(left=0.025, right=0.995, top=0.995, bottom=0.02)
    return fig


def build_exposure_figure() -> plt.Figure:
    fig = plt.figure(figsize=(7.2, 4.15), facecolor=WHITE)
    draw_panel_b(fig.add_subplot(1, 1, 1))
    fig.subplots_adjust(left=0.025, right=0.995, top=0.995, bottom=0.02)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for SVG, PDF, and PNG exports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = (
        ("instance_centric_workflow", build_overview_figure),
        ("exposure_target_construction", build_exposure_figure),
    )
    for name, builder in figures:
        stem = args.output_dir / name
        fig = builder()
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.035)
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
        fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.035)
        plt.close(fig)
        print(f"Wrote {stem}.svg")
        print(f"Wrote {stem}.pdf")
        print(f"Wrote {stem}.png")


if __name__ == "__main__":
    main()
