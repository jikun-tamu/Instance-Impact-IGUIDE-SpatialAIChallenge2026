#!/usr/bin/env python3
"""Fit source-aware Stage-2a population blends by classification source."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

try:
    from blend_stage2a_population import (
        _features,
        _geometry_predictions_for_predictions,
        _make_blended_rows,
        _prediction_vectors,
        _ridge_fit,
        _select_blend_weight,
        _standardize,
        _targets,
    )
    from scripts.stage2a.common import (
        DEFAULT_GEOMETRY_COLS,
        metrics_from_prediction_rows,
        parse_list,
        read_csv_rows,
        save_json,
        stage2a_prediction_fields,
        write_csv_rows,
    )
except ImportError:  # pragma: no cover
    from II_package.scripts.blend_stage2a_population import (
        _features,
        _geometry_predictions_for_predictions,
        _make_blended_rows,
        _prediction_vectors,
        _ridge_fit,
        _select_blend_weight,
        _standardize,
        _targets,
    )
    from II_package.scripts.stage2a.common import (
        DEFAULT_GEOMETRY_COLS,
        metrics_from_prediction_rows,
        parse_list,
        read_csv_rows,
        save_json,
        stage2a_prediction_fields,
        write_csv_rows,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit separate Stage-2a population blends by label source.")
    p.add_argument("--labels_csv", type=Path, required=True)
    p.add_argument("--split_manifest", type=Path, required=True)
    p.add_argument("--val_predictions", type=Path, required=True)
    p.add_argument("--test_predictions", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--source_col", type=str, default="classification_source")
    p.add_argument("--geometry_cols", type=str, default=",".join(DEFAULT_GEOMETRY_COLS))
    p.add_argument("--ridge_lambda", type=float, default=1.0)
    p.add_argument("--blend_grid_steps", type=int, default=101)
    p.add_argument("--min_train_rows", type=int, default=100)
    p.add_argument("--min_val_rows", type=int, default=30)
    p.add_argument(
        "--selection_metric",
        choices=["population_log_mae", "population_factor2_hit_rate"],
        default="population_log_mae",
    )
    return p.parse_args()


def _row_index(rows: Sequence[Mapping[str, object]]) -> Dict[str, Mapping[str, object]]:
    return {str(r.get("building_uid", "")): r for r in rows if str(r.get("building_uid", ""))}


def _load_manifest_indices(path: Path) -> List[int]:
    import json

    manifest = json.loads(path.read_text())
    return [int(i) for i in manifest["indices"]["train"]]


def _source_value(label: Mapping[str, object], source_col: str) -> str:
    value = str(label.get(source_col, "")).strip()
    return value if value else "__missing__"


def _filter_prediction_rows(
    pred_rows: Sequence[Mapping[str, object]],
    label_by_id: Mapping[str, Mapping[str, object]],
    source_col: str,
    source_value: str,
) -> List[Mapping[str, object]]:
    out = []
    for row in pred_rows:
        bid = str(row.get("building_uid", ""))
        label = label_by_id.get(bid)
        if label is not None and _source_value(label, source_col) == source_value:
            out.append(row)
    return out


def _fit_ridge(labels: Sequence[Mapping[str, object]], indices: Sequence[int], geometry_cols: Sequence[str], lam: float) -> Dict[str, object]:
    train_x = _features(labels, indices, geometry_cols)
    train_y = _targets(labels, indices)
    train_xs, _, mu, sd = _standardize(train_x, train_x)
    beta = _ridge_fit(train_xs, train_y, lam)
    return {"mu": mu, "sd": sd, "beta": beta, "train_rows": len(indices)}


def _fit_source_model(
    labels: Sequence[Mapping[str, object]],
    label_by_id: Mapping[str, Mapping[str, object]],
    train_idx: Sequence[int],
    val_rows_all: Sequence[Mapping[str, object]],
    geometry_cols: Sequence[str],
    source_col: str,
    source_value: str,
    global_model: Mapping[str, object],
    args: argparse.Namespace,
) -> Dict[str, object]:
    source_train_idx = [i for i in train_idx if _source_value(labels[i], source_col) == source_value]
    source_val_rows = _filter_prediction_rows(val_rows_all, label_by_id, source_col, source_value)
    enough = len(source_train_idx) >= args.min_train_rows and len(source_val_rows) >= args.min_val_rows
    model = _fit_ridge(labels, source_train_idx, geometry_cols, args.ridge_lambda) if enough else dict(global_model)
    val_neural, val_target, kept_val_rows = _prediction_vectors(source_val_rows, label_by_id)
    if len(kept_val_rows) == 0:
        selected = {"weight_neural": 0.0, "weight_geometry": 1.0, "validation_metric": float("nan")}
    else:
        val_geometry = _geometry_predictions_for_predictions(
            kept_val_rows,
            label_by_id,
            geometry_cols,
            np.asarray(model["mu"], dtype=np.float64),
            np.asarray(model["sd"], dtype=np.float64),
            np.asarray(model["beta"], dtype=np.float64),
        )
        selected = _select_blend_weight(val_neural, val_geometry, val_target, args.blend_grid_steps, args.selection_metric)
    return {
        "source": source_value,
        "used_source_specific_ridge": bool(enough),
        "train_rows": int(len(source_train_idx)),
        "val_rows": int(len(kept_val_rows)),
        "ridge": model,
        "blend": selected,
    }


def _select_global_fallback_blend(
    val_rows_all: Sequence[Mapping[str, object]],
    label_by_id: Mapping[str, Mapping[str, object]],
    geometry_cols: Sequence[str],
    global_model: Mapping[str, object],
    args: argparse.Namespace,
) -> Dict[str, float]:
    val_neural, val_target, kept_val_rows = _prediction_vectors(val_rows_all, label_by_id)
    val_geometry = _geometry_predictions_for_predictions(
        kept_val_rows,
        label_by_id,
        geometry_cols,
        np.asarray(global_model["mu"], dtype=np.float64),
        np.asarray(global_model["sd"], dtype=np.float64),
        np.asarray(global_model["beta"], dtype=np.float64),
    )
    return _select_blend_weight(val_neural, val_geometry, val_target, args.blend_grid_steps, args.selection_metric)


def _apply_source_models(
    test_rows_all: Sequence[Mapping[str, object]],
    label_by_id: Mapping[str, Mapping[str, object]],
    geometry_cols: Sequence[str],
    source_col: str,
    source_models: Mapping[str, Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, int]]:
    blended_rows: List[Dict[str, object]] = []
    geometry_rows: List[Dict[str, object]] = []
    test_counts: Dict[str, int] = {}
    for source_value, model in source_models.items():
        rows = _filter_prediction_rows(test_rows_all, label_by_id, source_col, source_value)
        test_counts[source_value] = len(rows)
        if not rows:
            continue
        neural_log, _, kept_rows = _prediction_vectors(rows, label_by_id)
        ridge = model["ridge"]
        geometry_log = _geometry_predictions_for_predictions(
            kept_rows,
            label_by_id,
            geometry_cols,
            np.asarray(ridge["mu"], dtype=np.float64),
            np.asarray(ridge["sd"], dtype=np.float64),
            np.asarray(ridge["beta"], dtype=np.float64),
        )
        blended_rows.extend(_make_blended_rows(kept_rows, label_by_id, neural_log, geometry_log, model["blend"]["weight_neural"]))
        geometry_rows.extend(_make_blended_rows(kept_rows, label_by_id, neural_log, geometry_log, 0.0))
    return blended_rows, geometry_rows, test_counts


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = read_csv_rows(args.labels_csv)
    label_by_id = _row_index(labels)
    train_idx = _load_manifest_indices(args.split_manifest)
    geometry_cols = parse_list(args.geometry_cols, DEFAULT_GEOMETRY_COLS)
    val_rows_all = read_csv_rows(args.val_predictions)
    test_rows_all = read_csv_rows(args.test_predictions)

    global_model = _fit_ridge(labels, train_idx, geometry_cols, args.ridge_lambda)
    global_fallback_blend = _select_global_fallback_blend(val_rows_all, label_by_id, geometry_cols, global_model, args)
    sources = sorted({_source_value(labels[i], args.source_col) for i in train_idx})
    sources.extend(sorted({_source_value(label_by_id[str(r.get("building_uid", ""))], args.source_col) for r in test_rows_all if str(r.get("building_uid", "")) in label_by_id} - set(sources)))
    source_models = {
        source: _fit_source_model(labels, label_by_id, train_idx, val_rows_all, geometry_cols, args.source_col, source, global_model, args)
        for source in sources
    }
    blended_rows, geometry_rows, test_counts = _apply_source_models(test_rows_all, label_by_id, geometry_cols, args.source_col, source_models)

    extra_fields = ["neural_pred_log1p_population", "geometry_pred_log1p_population"]
    fields = [*stage2a_prediction_fields(True), *extra_fields]
    write_csv_rows(args.out_dir / "test_predictions_source_aware_blended.csv", blended_rows, fields)
    write_csv_rows(args.out_dir / "test_predictions_source_aware_geometry.csv", geometry_rows, fields)

    metrics = {
        "source_col": args.source_col,
        "selection_metric": args.selection_metric,
        "geometry_cols": geometry_cols,
        "min_train_rows": args.min_train_rows,
        "min_val_rows": args.min_val_rows,
        "fallback_blend": global_fallback_blend,
        "source_models": {
            source: {
                "used_source_specific_ridge": model["used_source_specific_ridge"],
                "train_rows": model["train_rows"],
                "val_rows": model["val_rows"],
                "test_rows": test_counts.get(source, 0),
                "blend": model["blend"],
            }
            for source, model in source_models.items()
        },
        "source_aware_geometry_population_test_metrics": metrics_from_prediction_rows(geometry_rows),
        "source_aware_blended_test_metrics": metrics_from_prediction_rows(blended_rows),
    }
    save_json(args.out_dir / "source_aware_blend_metrics.json", metrics)
    save_json(
        args.out_dir / "source_aware_geometry_ridge_models.json",
        {
            "source_col": args.source_col,
            "geometry_cols": geometry_cols,
            "ridge_lambda": args.ridge_lambda,
            "fallback_model": {
                "feature_mean": np.asarray(global_model["mu"]).tolist(),
                "feature_std": np.asarray(global_model["sd"]).tolist(),
                "beta": np.asarray(global_model["beta"]).tolist(),
                "blend": global_fallback_blend,
            },
            "models": {
                source: {
                    "used_source_specific_ridge": model["used_source_specific_ridge"],
                    "feature_mean": np.asarray(model["ridge"]["mu"]).tolist(),
                    "feature_std": np.asarray(model["ridge"]["sd"]).tolist(),
                    "beta": np.asarray(model["ridge"]["beta"]).tolist(),
                    "blend": model["blend"],
                }
                for source, model in source_models.items()
            },
        },
    )
    print("[done] sources=", sorted(source_models))
    print("[done] metrics=", args.out_dir / "source_aware_blend_metrics.json")


if __name__ == "__main__":
    main()
