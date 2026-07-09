#!/usr/bin/env python3
"""Draw a Nature-style Stage2a training pipeline schematic.

Figure contract
---------------
Core conclusion:
    Stage2a training is a closed-world three-class proxy-label pipeline:
    label-policy locking and tile-blocked splits feed native type models,
    whose soft probabilities drive a deployable population regressor.
Figure archetype:
    Schematic-led composite.
Target journal/output:
    Nature-style double-column methods figure; SVG/PDF/TIFF plus PNG preview.
Backend:
    Python / matplotlib only.
Source traceability:
    Values are taken from docs/stage2a.md current Stage2b-facing decision
    notes and M31-M35 native three-class results.
Reviewer risk:
    Population labels are exposure proxies, not direct observed per-building
    population ground truth; the figure states this explicitly.
"""

from __future__ import annotations

import csv
from pathlib import Path
from textwrap import wrap

import matplotlib as mpl

mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "figures"
BASE = OUT_DIR / "stage2a_training_pipeline"


PALETTE = {
    "ink": "#272727",
    "muted": "#606060",
    "line": "#A8A8A8",
    "panel": "#F7F7F7",
    "blue": "#0F4D92",
    "blue_soft": "#DCEBFA",
    "teal": "#42949E",
    "teal_soft": "#DFF1F2",
    "violet": "#7C6CCF",
    "violet_soft": "#E8E4F7",
    "rose": "#B64342",
    "rose_soft": "#F6CFCB",
    "green": "#2E9E44",
    "green_soft": "#DDF3DE",
    "gold": "#C99321",
    "gold_soft": "#F4E5BF",
    "neutral": "#D8D8D8",
}


PIPELINE_STEPS = [
    {
        "title": "Proxy labels",
        "body": "Harris CBG exposure labels\ncrop + mask paths\ngeometry columns",
        "tag": "source",
        "color": PALETTE["blue_soft"],
        "edge": PALETTE["blue"],
    },
    {
        "title": "Policy gate",
        "body": "remove institutional/other\nactive: 3 classes\nm20 tile-blocked split",
        "tag": "lock",
        "color": PALETTE["teal_soft"],
        "edge": PALETTE["teal"],
    },
    {
        "title": "Native type",
        "body": "ConvNeXt-Tiny\nRGB + mask + geometry\n5 seeds, weighted CE",
        "tag": "GPU",
        "color": PALETTE["violet_soft"],
        "edge": PALETTE["violet"],
    },
    {
        "title": "OOF features",
        "body": "OOF train predictions\nvalidation probabilities\nensemble check",
        "tag": "OOF",
        "color": PALETTE["gold_soft"],
        "edge": PALETTE["gold"],
    },
    {
        "title": "Population model",
        "body": "log_footprint_m2\n+ soft type probabilities\n+ residual HGB",
        "tag": "proxy",
        "color": PALETTE["green_soft"],
        "edge": PALETTE["green"],
    },
    {
        "title": "Package",
        "body": "5 native checkpoints\npopulation model artifact\nStage2b exposure fields",
        "tag": "deploy",
        "color": PALETTE["rose_soft"],
        "edge": PALETTE["rose"],
    },
]

METRICS = [
    {
        "panel": "type",
        "label": "Native type macro-F1",
        "value": "0.8529 +/- 0.0106",
        "note": "M31, five seeds",
    },
    {
        "panel": "type",
        "label": "Native type accuracy",
        "value": "0.9300 +/- 0.0035",
        "note": "M31, five seeds",
    },
    {
        "panel": "population",
        "label": "Population log MAE",
        "value": "0.0760 +/- 0.0032",
        "note": "M32, five seeds",
    },
    {
        "panel": "ensemble",
        "label": "Ensemble log MAE",
        "value": "0.0734",
        "note": "M33 same-split check",
    },
]

TYPE_CONTRIBUTION = [
    {
        "model": "no type\nresidual HGB",
        "log_mae": 0.1157,
        "color": PALETTE["neutral"],
    },
    {
        "model": "soft native type\nresidual HGB",
        "log_mae": 0.0742,
        "color": PALETTE["green"],
    },
]


def apply_publication_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def write_source_data() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (BASE.with_name(BASE.name + "_source_data.csv")).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "label", "value", "note"])
        writer.writeheader()
        writer.writerow(
            {
                "section": "label policy",
                "label": "rows retained",
                "value": "21428 / 23014",
                "note": "drop institutional and other; retained fraction 0.9311",
            }
        )
        for metric in METRICS:
            writer.writerow(
                {
                    "section": metric["panel"],
                    "label": metric["label"],
                    "value": metric["value"],
                    "note": metric["note"],
                }
            )
        for row in TYPE_CONTRIBUTION:
            writer.writerow(
                {
                    "section": "type contribution",
                    "label": row["model"].replace("\n", " "),
                    "value": row["log_mae"],
                    "note": "M35 native population type-contribution check",
                }
            )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.025,
        1.025,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=PALETTE["ink"],
    )


def wrap_text(text: str, width: int) -> str:
    lines: list[str] = []
    for part in text.split("\n"):
        wrapped = wrap(part, width=width) or [""]
        lines.extend(wrapped)
    return "\n".join(lines)


def draw_round_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    tag: str,
    face: str,
    edge: str,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.028",
        linewidth=1.05,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.035,
        y + h - 0.065,
        title,
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    ax.text(
        x + 0.035,
        y + h - 0.16,
        wrap_text(body, 26),
        ha="left",
        va="top",
        fontsize=5.7,
        linespacing=1.25,
        color=PALETTE["muted"],
    )
    ax.text(
        x + w - 0.035,
        y + h - 0.065,
        tag,
        ha="right",
        va="top",
        fontsize=5.4,
        color=edge,
        fontweight="bold",
    )


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#808080") -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=1.0,
        color=color,
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(arrow)


def panel_a(ax: plt.Axes) -> None:
    ax.set_axis_off()
    add_panel_label(ax, "a")
    ax.text(
        0.0,
        0.99,
        "Stage2a native3 training pipeline",
        ha="left",
        va="top",
        fontsize=9.4,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    ax.text(
        0.0,
        0.925,
        "Closed-world three-class type training feeds a deployable population proxy regressor.",
        ha="left",
        va="top",
        fontsize=6.8,
        color=PALETTE["muted"],
    )

    positions = [
        (0.025, 0.57),
        (0.375, 0.57),
        (0.725, 0.57),
        (0.025, 0.235),
        (0.375, 0.235),
        (0.725, 0.235),
    ]
    w = 0.255
    h = 0.265
    for i, (step, (x, y)) in enumerate(zip(PIPELINE_STEPS, positions)):
        draw_round_box(
            ax,
            x,
            y,
            w,
            h,
            step["title"],
            step["body"],
            step["tag"],
            step["color"],
            step["edge"],
        )
    draw_arrow(ax, (positions[0][0] + w + 0.018, positions[0][1] + h * 0.52), (positions[1][0] - 0.018, positions[1][1] + h * 0.52))
    draw_arrow(ax, (positions[1][0] + w + 0.018, positions[1][1] + h * 0.52), (positions[2][0] - 0.018, positions[2][1] + h * 0.52))
    draw_arrow(ax, (positions[2][0] + w * 0.5, positions[2][1] - 0.02), (positions[3][0] + w * 0.5, positions[3][1] + h + 0.02))
    draw_arrow(ax, (positions[3][0] + w + 0.018, positions[3][1] + h * 0.52), (positions[4][0] - 0.018, positions[4][1] + h * 0.52))
    draw_arrow(ax, (positions[4][0] + w + 0.018, positions[4][1] + h * 0.52), (positions[5][0] - 0.018, positions[5][1] + h * 0.52))

    ax.text(
        0.015,
        0.085,
        "Training controls",
        ha="left",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    controls = [
        ("blocked split", "tile_base groups, no tile leakage"),
        ("selection", "macro-F1 for type, log MAE for population"),
        ("deployable features", "RGB, mask, log footprint, soft type"),
        ("limitation", "population label is an exposure proxy"),
    ]
    for i, (head, body) in enumerate(controls):
        x = 0.19 + i * 0.195
        ax.scatter([x], [0.085], s=20, color=PALETTE["ink"], zorder=3)
        ax.text(x + 0.018, 0.105, head, ha="left", va="center", fontsize=5.9, fontweight="bold", color=PALETTE["ink"])
        ax.text(x + 0.018, 0.06, wrap_text(body, 24), ha="left", va="center", fontsize=5.0, color=PALETTE["muted"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def panel_b(ax: plt.Axes) -> None:
    add_panel_label(ax, "b")
    total = 23014
    retained = 21428
    dropped = total - retained
    retained_frac = retained / total

    ax.set_xlim(0, total)
    ax.set_ylim(-0.05, 1.0)
    ax.barh([0.55], [retained], height=0.24, color=PALETTE["teal"], edgecolor=PALETTE["ink"], linewidth=0.4)
    ax.barh([0.55], [dropped], left=[retained], height=0.24, color=PALETTE["rose_soft"], edgecolor=PALETTE["ink"], linewidth=0.4, hatch="//")
    ax.text(0, 0.92, "Label-policy gate", ha="left", va="top", fontsize=8.4, fontweight="bold", color=PALETTE["ink"])
    ax.text(
        0,
        0.79,
        "Drop institutional/other to train a cleaner native three-class task.",
        ha="left",
        va="top",
        fontsize=6.3,
        color=PALETTE["muted"],
    )
    ax.text(retained * 0.5, 0.55, "retained\n21,428", ha="center", va="center", fontsize=6.6, color="white", fontweight="bold")
    ax.text(retained + dropped * 0.5, 0.55, "dropped\n1,586", ha="center", va="center", fontsize=6.1, color=PALETTE["rose"])
    ax.text(
        total * 0.5,
        0.23,
        f"{retained_frac:.1%} retained; metrics over three active classes.",
        ha="center",
        va="center",
        fontsize=6.0,
        color=PALETTE["muted"],
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_c(ax: plt.Axes) -> None:
    add_panel_label(ax, "c")
    ax.set_axis_off()
    ax.text(
        0.0,
        0.96,
        "Training evidence carried into the figure",
        ha="left",
        va="top",
        fontsize=8.4,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    x0 = 0.0
    card_w = 0.445
    card_h = 0.17
    gap_x = 0.045
    gap_y = 0.035
    card_colors = [PALETTE["violet_soft"], PALETTE["blue_soft"], PALETTE["green_soft"], PALETTE["gold_soft"]]
    edge_colors = [PALETTE["violet"], PALETTE["blue"], PALETTE["green"], PALETTE["gold"]]
    for i, metric in enumerate(METRICS):
        row = i // 2
        col = i % 2
        x = x0 + col * (card_w + gap_x)
        y0 = 0.66 - row * (card_h + gap_y)
        card = FancyBboxPatch(
            (x, y0),
            card_w,
            card_h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor=card_colors[i],
            edgecolor=edge_colors[i],
            linewidth=0.8,
        )
        ax.add_patch(card)
        ax.text(x + 0.02, y0 + card_h - 0.034, metric["label"], ha="left", va="top", fontsize=5.8, color=PALETTE["muted"])
        ax.text(x + 0.02, y0 + card_h - 0.092, metric["value"], ha="left", va="top", fontsize=6.4, fontweight="bold", color=PALETTE["ink"])
        ax.text(x + card_w - 0.02, y0 + 0.025, metric["note"].split(",")[0], ha="right", va="bottom", fontsize=5.0, color=PALETTE["muted"])

    # Small quantitative support bar: predicted type contribution to population.
    bar_x = 0.045
    bar_y = 0.115
    bar_w = 0.36
    max_val = 0.13
    ax.text(
        bar_x,
        0.315,
        "Soft native type improves population proxy error",
        ha="left",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    for i, row in enumerate(TYPE_CONTRIBUTION):
        y = bar_y + (1 - i) * 0.11
        width = bar_w * row["log_mae"] / max_val
        ax.add_patch(Rectangle((bar_x, y), width, 0.055, facecolor=row["color"], edgecolor=PALETTE["ink"], linewidth=0.35))
        ax.text(bar_x + width + 0.012, y + 0.028, f"{row['log_mae']:.4f}", ha="left", va="center", fontsize=6.2, color=PALETTE["ink"])
        ax.text(bar_x - 0.012, y + 0.028, row["model"], ha="right", va="center", fontsize=5.6, color=PALETTE["muted"])
    ax.text(bar_x, bar_y - 0.055, "Validation log MAE; lower is better", ha="left", va="center", fontsize=5.5, color=PALETTE["muted"])

    # Export bundle capsule.
    capsule = FancyBboxPatch(
        (0.55, 0.06),
        0.41,
        0.255,
        boxstyle="round,pad=0.016,rounding_size=0.025",
        facecolor=PALETTE["panel"],
        edgecolor=PALETTE["line"],
        linewidth=0.75,
    )
    ax.add_patch(capsule)
    ax.text(0.575, 0.265, "Expected training outputs", ha="left", va="center", fontsize=6.8, fontweight="bold", color=PALETTE["ink"])
    ax.text(
        0.575,
        0.19,
        "checkpoint + val predictions\npopulation artifact + metrics",
        ha="left",
        va="top",
        fontsize=5.8,
        linespacing=1.35,
        color=PALETTE["muted"],
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def build_figure() -> plt.Figure:
    apply_publication_style()
    fig = plt.figure(figsize=(7.2, 5.75))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.85, 1.25],
        width_ratios=[1.0, 1.45],
        hspace=0.28,
        wspace=0.24,
        left=0.055,
        right=0.985,
        top=0.95,
        bottom=0.08,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)
    return fig


def save_outputs(fig: plt.Figure) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(BASE.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(BASE.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(BASE.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(BASE.with_suffix(".png"), dpi=300, bbox_inches="tight")


def main() -> None:
    write_source_data()
    fig = build_figure()
    save_outputs(fig)
    print(f"wrote {BASE.with_suffix('.svg')}")
    print(f"wrote {BASE.with_suffix('.pdf')}")
    print(f"wrote {BASE.with_suffix('.tiff')}")
    print(f"wrote {BASE.with_suffix('.png')}")
    print(f"wrote {BASE.with_name(BASE.name + '_source_data.csv')}")


if __name__ == "__main__":
    main()
