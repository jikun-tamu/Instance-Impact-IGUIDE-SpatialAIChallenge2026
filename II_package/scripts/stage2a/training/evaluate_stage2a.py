#!/usr/bin/env python3
"""Evaluate a Stage-2a checkpoint on a locked split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from scripts.stage2a.common import (
        DEFAULT_GEOMETRY_COLS,
        Stage2aDataset,
        apply_population_blend,
        build_model_from_config,
        class_names_from_config,
        collate_stage2a,
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
        write_csv_rows,
    )
except ImportError:  # pragma: no cover
    from II_package.scripts.stage2a.common import (
        DEFAULT_GEOMETRY_COLS,
        Stage2aDataset,
        apply_population_blend,
        build_model_from_config,
        class_names_from_config,
        collate_stage2a,
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
        write_csv_rows,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Stage-2a checkpoint.")
    p.add_argument("--labels_csv", type=Path, required=True)
    p.add_argument("--split_manifest", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--out_metrics", type=Path, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--population_blend_json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv_rows(args.labels_csv)
    manifest = load_json(args.split_manifest)
    indices = manifest["indices"][args.split]
    config = load_ckpt_config(args.ckpt)
    config.setdefault("input_mode", config.get("input_mode", "rgb_mask"))
    config.setdefault("pooling_mode", config.get("pooling_mode", "global"))
    config.setdefault("geometry_cols", config.get("geometry_cols", DEFAULT_GEOMETRY_COLS))
    class_names = class_names_from_config(config)
    geometry_cols = parse_list(config.get("geometry_cols"), DEFAULT_GEOMETRY_COLS)
    blend_model = None
    if args.population_blend_json is not None:
        blend_model = load_json(args.population_blend_json)
        blend_geometry_cols = parse_list(blend_model.get("geometry_cols"), DEFAULT_GEOMETRY_COLS)
        if blend_geometry_cols != geometry_cols:
            raise ValueError(f"Blend geometry cols {blend_geometry_cols} do not match checkpoint geometry cols {geometry_cols}")
    ds = Stage2aDataset(
        rows,
        indices=indices,
        img_size=int(config.get("img_size", args.img_size)),
        input_mode=str(config.get("input_mode", "rgb_mask")),
        geometry_cols=geometry_cols,
        ring_mask_col=str(config.get("ring_mask_col", "mask_R")),
        ring_radius_px=int(config.get("ring_radius_px", 48)),
        require_ring_mask=str(config.get("pooling_mode", "global")) == "mask_m_ring" or "ring" in str(config.get("input_mode", "rgb_mask")),
        train=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_stage2a)
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = build_model_from_config(config, pretrained=False).to(device)
    model.load_state_dict(load_state_dict_from_ckpt(args.ckpt), strict=True)
    model.eval()
    pred_rows = []
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["x"].to(device),
                geometry=batch["geometry"].to(device),
                mask=batch["mask"].to(device),
                ring_mask=batch["ring_mask"].to(device),
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
            rows_batch = prediction_rows_from_outputs(
                batch["meta"],
                pred_pop_log,
                out["type_probs"].detach().cpu().numpy(),
                batch["pop_log"].numpy(),
                batch["type_idx"].numpy(),
                class_names=class_names,
            )
            if blend_model is not None:
                for i, row in enumerate(rows_batch):
                    row["neural_pred_log1p_population"] = float(neural_pop_log[i])
                    row["geometry_pred_log1p_population"] = float(geometry_pop_log[i])
                    row.update(blend_meta[i])
            pred_rows.extend(rows_batch)
    metrics = metrics_from_prediction_rows(pred_rows, class_names=class_names)
    save_json(args.out_metrics, metrics)
    fields = stage2a_prediction_fields(True, class_names=class_names)
    if blend_model is not None:
        fields = [
            *fields,
            "neural_pred_log1p_population",
            "geometry_pred_log1p_population",
            "population_blend_source",
            "population_blend_input_source",
            "population_blend_used_fallback",
            "population_blend_weight_neural",
            "population_blend_weight_geometry",
        ]
    write_csv_rows(args.out_csv, pred_rows, fields)
    print("[done] rows=", len(pred_rows))
    print("[done] wrote=", args.out_metrics)


if __name__ == "__main__":
    main()
