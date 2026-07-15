#!/usr/bin/env python3
"""Build the empirical plate linking building evidence to regional M2b output.

The Stage 2a example is a separate Harris County validation record. The LA
panels use only the authoritative coverage-aware M2b fields and East map.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import zipfile
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.titleweight": "bold",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "instance-impact-evidence-plate",
    }
)

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle
from PIL import Image
from rasterio.warp import transform_geom


INK = "#17212B"
MUTED = "#66717D"
WHITE = "#FFFFFF"
DAMAGE_COLORS = {
    0: "#2ecc71",
    1: "#f39c12",
    2: "#e67e22",
    3: "#e74c3c",
    -1: "#95a5a6",
}


def exterior_rings(geometry: dict) -> list[list[list[float]]]:
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"][0]]
    if geometry["type"] == "MultiPolygon":
        return [polygon[0] for polygon in geometry["coordinates"]]
    raise ValueError(f"Unsupported geometry type: {geometry['type']}")


def ring_area(ring: list[list[float]]) -> float:
    return abs(
        sum(
            x0 * y1 - x1 * y0
            for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1])
        )
    ) / 2


def geometry_bounds(geometry: dict) -> tuple[float, float, float, float]:
    points = [point for ring in exterior_rings(geometry) for point in ring]
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)


def load_stage2a_example(csv_path: Path, archive_path: Path) -> tuple[dict, Image.Image]:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["true_type_class"] == "residential_multi"
            and row["pred_type_class"] == row["true_type_class"]
        ]
    if not rows:
        raise RuntimeError("No correctly classified residential-multi validation rows")

    for row in rows:
        row["log_error"] = abs(
            float(row["true_log1p_population"])
            - float(row["pred_log1p_population"])
        )
    median_error = statistics.median(row["log_error"] for row in rows)
    median_conf = statistics.median(float(row["pred_type_conf"]) for row in rows)
    error_scale = statistics.pstdev(row["log_error"] for row in rows) or 1.0
    conf_scale = statistics.pstdev(float(row["pred_type_conf"]) for row in rows) or 1.0
    for row in rows:
        row["selection_score"] = (
            abs(row["log_error"] - median_error) / error_scale
            + abs(float(row["pred_type_conf"]) - median_conf) / conf_scale
        )

    # Select for statistical typicality first, then legibility of the archived crop.
    candidates = sorted(rows, key=lambda row: (row["selection_score"], row["building_uid"]))[:32]
    with zipfile.ZipFile(archive_path) as archive:
        readable: list[tuple[int, str, dict, Image.Image]] = []
        for row in candidates:
            member = f"ml_dataset/crops/{Path(row['crop_path']).name}"
            image = Image.open(io.BytesIO(archive.read(member))).convert("RGB")
            readable.append((image.width * image.height, row["building_uid"], row, image.copy()))
    _, _, selected, crop = max(readable, key=lambda item: (item[0], item[1]))
    return selected, crop


def load_rgb(path: Path) -> tuple[np.ndarray, object, object]:
    with rasterio.open(path) as dataset:
        rgb = np.moveaxis(dataset.read((1, 2, 3)), 0, -1)
        return rgb, dataset.crs, dataset.bounds


def load_cell_features(path: Path, dst_crs: object) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    features = []
    for feature in data["features"]:
        if feature["properties"].get("cell_id") != "cell_00507":
            continue
        features.append(
            {
                "properties": feature["properties"],
                "geometry": transform_geom(
                    "EPSG:4326", dst_crs, feature["geometry"], precision=3
                ),
            }
        )
    if not features:
        raise RuntimeError("No cell_00507 features found")
    return features


def select_la_example(features: list[dict], raster_bounds: object) -> dict:
    candidates = []
    for feature in features:
        props = feature["properties"]
        bounds = geometry_bounds(feature["geometry"])
        inside = (
            bounds[0] > raster_bounds.left + 50
            and bounds[1] > raster_bounds.bottom + 50
            and bounds[2] < raster_bounds.right - 50
            and bounds[3] < raster_bounds.top - 50
        )
        if (
            props.get("m2b_damage_class") == 1
            and props.get("n_dates_valid_coverage") == props.get("n_dates_total") == 4
            and not props.get("is_unstable")
            and float(props.get("sam3_confidence") or 0) >= 0.75
            and inside
        ):
            area = sum(ring_area(ring) for ring in exterior_rings(feature["geometry"]))
            candidates.append((area, props["bldg_uid"], feature))
    if not candidates:
        raise RuntimeError("No stable, fully observed minor-damage example found")
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def panel_title(axis: plt.Axes, letter: str, title: str, subtitle: str = "") -> None:
    axis.text(
        0,
        1.0,
        letter,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        0.075,
        1.0,
        title,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color=INK,
    )
    if subtitle:
        axis.text(
            0.075,
            0.90,
            subtitle,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=6.2,
            color=MUTED,
        )


def draw(
    output_dir: Path,
    *,
    stage2a_csv: Path,
    stage2a_archive: Path,
    pre_tif: Path,
    post_tif: Path,
    cell_geojson: Path,
    east_map: Path,
) -> None:
    stage2a_row, stage2a_crop = load_stage2a_example(stage2a_csv, stage2a_archive)
    pre_rgb, pre_crs, pre_bounds = load_rgb(pre_tif)
    post_rgb, post_crs, post_bounds = load_rgb(post_tif)
    if pre_crs != post_crs or pre_bounds != post_bounds:
        raise RuntimeError("Pre- and post-event rasters do not share a grid")
    features = load_cell_features(cell_geojson, post_crs)
    la_example = select_la_example(features, post_bounds)

    fig = plt.figure(figsize=(183 / 25.4, 150 / 25.4), facecolor=WHITE)
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=(0.36, 0.64),
        left=0.018,
        right=0.992,
        bottom=0.018,
        top=0.985,
        hspace=0.105,
    )
    top = grid[0].subgridspec(1, 3, width_ratios=(1.12, 1.22, 1.58), wspace=0.085)

    # a: Stage 2a is deliberately separate from the LA application.
    ax_a = fig.add_subplot(top[0])
    ax_a.axis("off")
    panel_title(
        ax_a,
        "a",
        "Stage 2a exposure proxy",
        "Harris County validation (separate study area)",
    )
    crop_ax = ax_a.inset_axes((0.00, 0.08, 0.48, 0.70))
    crop_ax.imshow(stage2a_crop, interpolation="nearest")
    crop_ax.set_xticks([])
    crop_ax.set_yticks([])
    for spine in crop_ax.spines.values():
        spine.set_color("#C4CBD2")
        spine.set_linewidth(0.6)
    record_ax = ax_a.inset_axes((0.52, 0.06, 0.48, 0.73))
    record_ax.axis("off")
    record_ax.text(0, 0.94, "Building type", fontsize=6.1, color=MUTED, va="top")
    record_ax.text(
        0,
        0.79,
        f"Residential multi\n({float(stage2a_row['pred_type_conf']):.3f})",
        fontsize=7.0,
        color=INK,
        fontweight="bold",
        va="top",
        linespacing=1.18,
    )
    record_ax.text(0, 0.48, "Constructed exposure proxy", fontsize=6.1, color=MUTED, va="top")
    record_ax.text(
        0,
        0.34,
        f"{float(stage2a_row['true_population']):.1f}",
        fontsize=7.0,
        color=INK,
        fontweight="bold",
        va="top",
    )
    record_ax.text(0, 0.16, "Model estimate", fontsize=6.1, color=MUTED, va="top")
    record_ax.text(
        0,
        0.02,
        f"{float(stage2a_row['pred_population']):.1f}",
        fontsize=7.0,
        color=INK,
        fontweight="bold",
        va="top",
    )

    # b: the same persistent polygon is shown in the paired observations.
    ax_b = fig.add_subplot(top[1])
    ax_b.axis("off")
    panel_title(ax_b, "b", "Persistent building evidence", "LA cell 00507")
    example_bounds = geometry_bounds(la_example["geometry"])
    pad = 30.0
    view_bounds = (
        example_bounds[0] - pad,
        example_bounds[1] - pad,
        example_bounds[2] + pad,
        example_bounds[3] + pad,
    )
    for index, (label, image) in enumerate(
        (("Pre-event", pre_rgb), ("Post-event", post_rgb))
    ):
        image_ax = ax_b.inset_axes((index * 0.51, 0.20, 0.49, 0.61))
        image_ax.imshow(
            image,
            extent=(post_bounds.left, post_bounds.right, post_bounds.bottom, post_bounds.top),
        )
        for ring in exterior_rings(la_example["geometry"]):
            xs, ys = zip(*ring)
            image_ax.plot(xs, ys, color=WHITE, linewidth=1.8)
            image_ax.plot(xs, ys, color=DAMAGE_COLORS[1], linewidth=1.0)
        image_ax.set_xlim(view_bounds[0], view_bounds[2])
        image_ax.set_ylim(view_bounds[1], view_bounds[3])
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        image_ax.set_title(label, fontsize=6.2, fontweight="normal", pad=2, color=INK)
        for spine in image_ax.spines.values():
            spine.set_color("#C4CBD2")
            spine.set_linewidth(0.6)
    props = la_example["properties"]
    ax_b.text(
        0.5,
        0.075,
        f"M2b: minor  |  {props['n_dates_valid_coverage']}/{props['n_dates_total']} valid dates  |  stable",
        transform=ax_b.transAxes,
        ha="center",
        va="center",
        fontsize=6.2,
        color=INK,
    )

    # c: all current M2b outputs in the 600 m cell, without debug labels.
    ax_c = fig.add_subplot(top[2])
    ax_c.axis("off")
    panel_title(ax_c, "c", "Building outputs within one 600 m cell")
    cell_ax = ax_c.inset_axes((0.00, 0.02, 1.00, 0.85))
    cell_ax.imshow(
        post_rgb,
        extent=(post_bounds.left, post_bounds.right, post_bounds.bottom, post_bounds.top),
    )
    grouped: dict[int, list[MplPolygon]] = {key: [] for key in DAMAGE_COLORS}
    for feature in features:
        damage_class = int(feature["properties"].get("m2b_damage_class", -1))
        for ring in exterior_rings(feature["geometry"]):
            grouped[damage_class].append(MplPolygon(ring, closed=True))
    for damage_class, patches in grouped.items():
        if not patches:
            continue
        cell_ax.add_collection(
            PatchCollection(
                patches,
                facecolor=DAMAGE_COLORS[damage_class],
                edgecolor=WHITE,
                linewidth=0.18,
                alpha=0.58,
            )
        )
    cell_ax.add_patch(
        Rectangle(
            (view_bounds[0], view_bounds[1]),
            view_bounds[2] - view_bounds[0],
            view_bounds[3] - view_bounds[1],
            fill=False,
            edgecolor=WHITE,
            linewidth=0.8,
            linestyle=(0, (2, 1)),
        )
    )
    cell_ax.set_xlim(post_bounds.left, post_bounds.right)
    cell_ax.set_ylim(post_bounds.bottom, post_bounds.top)
    cell_ax.set_xticks([])
    cell_ax.set_yticks([])
    for spine in cell_ax.spines.values():
        spine.set_color("#C4CBD2")
        spine.set_linewidth(0.6)

    # d: crop the authoritative East M2b PNG; retain its original class legend.
    ax_d = fig.add_subplot(grid[1])
    ax_d.axis("off")
    panel_title(
        ax_d,
        "d",
        "Coverage-aware regional output",
        "LA East cluster; M2b majority vote across valid post-event dates",
    )
    regional = Image.open(east_map).convert("RGB")
    width, height = regional.size
    regional = regional.crop((0, round(0.27 * height), round(0.985 * width), round(0.995 * height)))
    map_ax = ax_d.inset_axes((0.00, 0.00, 1.00, 0.87))
    map_ax.imshow(regional)
    map_ax.axis("off")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "instance_evidence_to_regional_output"
    fig.savefig(stem.with_suffix(".svg"), facecolor=WHITE, metadata={"Date": None})
    fig.savefig(
        stem.with_suffix(".pdf"),
        facecolor=WHITE,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=300,
        facecolor=WHITE,
        metadata={"Software": None},
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--stage2a-csv",
        type=Path,
        default=repo
        / "II_package/outputs/stage2a/population_type_features/"
        "m41_native3_split_sensitivity/m33_native3_calibration/"
        "raw_m33_ensemble_val_predictions.csv",
    )
    parser.add_argument(
        "--stage2a-archive",
        type=Path,
        default=repo / "stage2a_old_training_data_codes/Harris_CBG_Building_Population.zip",
    )
    parser.add_argument(
        "--pre-tif",
        type=Path,
        default=repo / "II_package/la_fire_results/demo_cell_00507/cell_00507_pre.tif",
    )
    parser.add_argument(
        "--post-tif",
        type=Path,
        default=repo
        / "II_package/la_fire_results/demo_cell_00507/cell_00507_post_20250110.tif",
    )
    parser.add_argument(
        "--cell-geojson",
        type=Path,
        default=repo / "II_package/la_fire_results/sample_cells.geojson",
    )
    parser.add_argument(
        "--east-map",
        type=Path,
        default=repo / "II_package/la_fire_results/maps/building_damage_map_east.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    draw(
        args.out_dir,
        stage2a_csv=args.stage2a_csv,
        stage2a_archive=args.stage2a_archive,
        pre_tif=args.pre_tif,
        post_tif=args.post_tif,
        cell_geojson=args.cell_geojson,
        east_map=args.east_map,
    )


if __name__ == "__main__":
    main()
