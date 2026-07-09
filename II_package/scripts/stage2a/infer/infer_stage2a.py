#!/usr/bin/env python3
"""Standalone Stage-2a inference over per-building crops and masks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from scripts.stage2a.common import (
        CLASS_NAMES,
        DEFAULT_GEOMETRY_COLS,
        MASK_GEOMETRY_COLS,
        PROB_COLUMNS,
        Stage2aInferenceDataset,
        apply_population_blend,
        build_model_from_config,
        class_names_from_config,
        collate_stage2a,
        has_value,
        load_ckpt_config,
        load_json,
        load_state_dict_from_ckpt,
        metrics_from_prediction_rows,
        parse_list,
        population_blend_metadata,
        prediction_rows_from_outputs,
        read_csv_rows,
        save_json,
        stage2a_prediction_fields,
    )
except ImportError:  # pragma: no cover
    from II_package.scripts.stage2a.common import (
        CLASS_NAMES,
        DEFAULT_GEOMETRY_COLS,
        MASK_GEOMETRY_COLS,
        PROB_COLUMNS,
        Stage2aInferenceDataset,
        apply_population_blend,
        build_model_from_config,
        class_names_from_config,
        collate_stage2a,
        has_value,
        load_ckpt_config,
        load_json,
        load_state_dict_from_ckpt,
        metrics_from_prediction_rows,
        parse_list,
        population_blend_metadata,
        prediction_rows_from_outputs,
        read_csv_rows,
        save_json,
        stage2a_prediction_fields,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Run Stage-2a model inference.")
    p.add_argument("--input_csv", type=Path, required=True, help="CSV with building_uid,crop_path,mask_path")
    p.add_argument("--ckpt", type=Path, required=True, help="Path to Stage-2a checkpoint")
    p.add_argument("--out_csv", type=Path, required=True, help="Output CSV path")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--limit", type=int, default=0, help="Optional row limit")
    p.add_argument("--print_examples", type=int, default=10)
    p.add_argument("--log_every_steps", type=int, default=20)
    p.add_argument("--crop_col", type=str, default="crop_path")
    p.add_argument("--mask_col", type=str, default="mask_path")
    p.add_argument("--id_col", type=str, default="building_uid")
    p.add_argument("--geometry_cols", type=str, default="", help="Optional geometry columns for refined Stage-2a checkpoints")
    p.add_argument("--population_blend_json", type=Path, default=None, help="Optional geometry ridge/blend artifact JSON")
    p.add_argument("--out_metrics", type=Path, default=None, help="Optional metrics JSON when input rows include truth")
    p.add_argument(
        "--allow_missing_geometry",
        action="store_true",
        help="Allow non-mask geometry inputs to fall back to missing indicators/zeros instead of failing fast",
    )
    return p.parse_args()


def _row_has_log_footprint(row: Mapping[str, object]) -> bool:
    return has_value(row.get("log_footprint_m2")) or has_value(row.get("footprint_m2"))


def _assert_deploy_geometry(rows: Sequence[Mapping[str, object]], geometry_cols: Sequence[str], allow_missing: bool) -> None:
    if allow_missing:
        return
    problems = []
    cols = list(geometry_cols)
    if "log_footprint_m2" in cols:
        missing = sum(1 for row in rows if not _row_has_log_footprint(row))
        if missing:
            problems.append(f"log_footprint_m2/footprint_m2 missing in {missing}/{len(rows)} rows")
    for col in cols:
        if col in MASK_GEOMETRY_COLS or col in {"log_footprint_m2", "estimated_units_missing", "GEOID_missing"}:
            continue
        missing = sum(1 for row in rows if not has_value(row.get(col)))
        if missing:
            problems.append(f"{col} missing in {missing}/{len(rows)} rows")
    if problems:
        msg = "; ".join(problems)
        raise ValueError(
            "Geometry checkpoint/blend requires deploy-visible geometry columns: "
            f"{msg}. Rebuild the inference CSV with geometry passthrough or pass --allow_missing_geometry for an explicit fallback."
        )


def main():
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {args.input_csv}")
    if not args.ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.ckpt}")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(args.input_csv, limit=args.limit)
    if len(rows) == 0:
        raise RuntimeError("No rows loaded from input CSV.")
    for c in [args.id_col, args.crop_col, args.mask_col]:
        if c not in rows[0]:
            raise KeyError(f"Missing required column '{c}' in input CSV.")

    config = load_ckpt_config(args.ckpt)
    config.setdefault("input_mode", "rgb_mask")
    config.setdefault("pooling_mode", "global")
    if args.geometry_cols:
        config["geometry_cols"] = args.geometry_cols
    else:
        config.setdefault("geometry_cols", DEFAULT_GEOMETRY_COLS)
    class_names = class_names_from_config(config)
    geometry_cols = parse_list(config.get("geometry_cols"), DEFAULT_GEOMETRY_COLS)

    blend_model = None
    if args.population_blend_json is not None:
        blend_model = load_json(args.population_blend_json)
        blend_geometry_cols = parse_list(blend_model.get("geometry_cols"), DEFAULT_GEOMETRY_COLS)
        if blend_geometry_cols != geometry_cols:
            raise ValueError(
                "population_blend_json geometry_cols must match the checkpoint geometry_cols for packaged inference. "
                f"blend={blend_geometry_cols}, checkpoint={geometry_cols}"
            )
    if "geometry" in str(config.get("input_mode", "rgb_mask")) or blend_model is not None:
        _assert_deploy_geometry(rows, geometry_cols, args.allow_missing_geometry)

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("[info] rows:", len(rows))
    print("[info] device:", device)
    print("[info] ckpt:", args.ckpt)

    ds = Stage2aInferenceDataset(
        rows,
        img_size=args.img_size,
        crop_col=args.crop_col,
        mask_col=args.mask_col,
        id_col=args.id_col,
        input_mode=str(config.get("input_mode", "rgb_mask")),
        geometry_cols=geometry_cols,
        ring_mask_col=str(config.get("ring_mask_col", "mask_R")),
        ring_radius_px=int(config.get("ring_radius_px", 48)),
        require_ring_mask=str(config.get("pooling_mode", "global")) == "mask_m_ring" or "ring" in str(config.get("input_mode", "rgb_mask")),
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        collate_fn=collate_stage2a,
    )

    model = build_model_from_config(config, pretrained=False).to(device)
    state = load_state_dict_from_ckpt(args.ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    out_rows = []
    step = 0
    with torch.no_grad():
        for batch in loader:
            step += 1
            out = model(
                batch["x"].to(device, non_blocking=use_cuda),
                geometry=batch["geometry"].to(device, non_blocking=use_cuda),
                mask=batch["mask"].to(device, non_blocking=use_cuda),
                ring_mask=batch["ring_mask"].to(device, non_blocking=use_cuda),
                return_dict=True,
            )
            neural_pop_log = out["pop_log"].detach().cpu().numpy()
            pred_pop_log = neural_pop_log
            geometry_pop_log = None
            blend_meta = None
            if blend_model is not None:
                source_values = [meta.get("classification_source", "") for meta in batch["meta"]]
                pred_pop_log, geometry_pop_log = apply_population_blend(
                    neural_pop_log,
                    batch["geometry"].detach().cpu().numpy(),
                    blend_model,
                    source_values=source_values,
                )
                blend_meta = population_blend_metadata(blend_model, source_values, n_rows=len(neural_pop_log))
            y_pop_log = batch["pop_log"].numpy() if all(meta.get("has_population_target") for meta in batch["meta"]) else None
            y_type = batch["type_idx"].numpy() if all(meta.get("has_type_target") for meta in batch["meta"]) else None
            pred_rows = prediction_rows_from_outputs(
                batch["meta"],
                pred_pop_log,
                out["type_probs"].detach().cpu().numpy(),
                y_pop_log=y_pop_log,
                y_type=y_type,
                class_names=class_names,
            )
            for i, (row, meta) in enumerate(zip(pred_rows, batch["meta"])):
                row["tile_id"] = meta.get("tile_id", meta.get("tile_base", ""))
                row["event_id"] = meta.get("event_id", "")
                row["sam3_confidence"] = meta.get("sam3_confidence", "")
                row["hazard_type"] = meta.get("hazard_type", "")
                if blend_model is not None:
                    row["neural_pred_log1p_population"] = float(neural_pop_log[i])
                    row["geometry_pred_log1p_population"] = float(geometry_pop_log[i])
                    row.update(blend_meta[i])
                out_rows.append(row)
            if step % max(1, args.log_every_steps) == 0:
                print(f"[infer_stage2a] step={step}/{len(loader)} done_rows={len(out_rows)}")

    include_truth = bool(out_rows) and all("true_population" in row for row in out_rows)
    fields = [
        *stage2a_prediction_fields(include_truth=include_truth, class_names=class_names),
        "tile_id",
        "event_id",
        "sam3_confidence",
        "hazard_type",
    ]
    if blend_model is not None:
        fields.extend(
            [
                "neural_pred_log1p_population",
                "geometry_pred_log1p_population",
                "population_blend_source",
                "population_blend_input_source",
                "population_blend_used_fallback",
                "population_blend_weight_neural",
                "population_blend_weight_geometry",
            ]
        )
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)
    if args.out_metrics is not None:
        if not include_truth:
            raise ValueError("--out_metrics requires input rows with estimated_population or true_population truth.")
        save_json(args.out_metrics, metrics_from_prediction_rows(out_rows, class_names=class_names))

    print("[done] rows:", len(out_rows))
    print("[done] wrote:", args.out_csv)
    if args.out_metrics is not None:
        print("[done] wrote:", args.out_metrics)
    counts = {}
    for r in out_rows:
        counts[r["pred_type_class"]] = counts.get(r["pred_type_class"], 0) + 1
    print("[summary] class_counts:", counts)
    pops = [float(r["pred_population"]) for r in out_rows]
    if pops:
        q = np.percentile(np.asarray(pops, dtype=np.float64), [0, 25, 50, 75, 95, 99])
        print("[summary] pred_population quantiles p0/p25/p50/p75/p95/p99:", [round(float(x), 3) for x in q])
    for r in out_rows[: max(0, args.print_examples)]:
        print(
            "[example]",
            r["building_uid"],
            r["pred_type_class"],
            round(float(r["pred_type_conf"]), 3),
            round(float(r["pred_population"]), 3),
        )


if __name__ == "__main__":
    main()
