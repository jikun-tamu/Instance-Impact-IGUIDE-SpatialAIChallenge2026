#!/usr/bin/env python3
"""Validation-only class-bias calibration for Stage-2a type predictions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

try:
    from scripts.stage2a.common import (
        CLASS_NAMES,
        PROB_COLUMNS,
        classification_metrics,
        metrics_from_prediction_rows,
        read_csv_rows,
        safe_float,
        safe_int,
        save_json,
        stage2a_prediction_fields,
        write_csv_rows,
    )
except ImportError:  # pragma: no cover
    from II_package.scripts.stage2a.common import (
        CLASS_NAMES,
        PROB_COLUMNS,
        classification_metrics,
        metrics_from_prediction_rows,
        read_csv_rows,
        safe_float,
        safe_int,
        save_json,
        stage2a_prediction_fields,
        write_csv_rows,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run validation-only class-bias calibration for Stage-2a type models.")
    p.add_argument("--experiment_root", type=Path, required=True, help="Directory containing M10 experiment subdirectories.")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--model_glob", type=str, default="m10_*_seed2025")
    p.add_argument("--bias_min", type=float, default=-4.0)
    p.add_argument("--bias_max", type=float, default=4.0)
    p.add_argument("--bias_step", type=float, default=0.25)
    p.add_argument("--passes", type=int, default=4)
    p.add_argument("--min_source_val_rows", type=int, default=20)
    return p.parse_args()


def _probs(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    return np.asarray([[safe_float(r.get(c), 0.0) for c in PROB_COLUMNS] for r in rows], dtype=np.float64)


def _labels(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    return np.asarray([safe_int(r.get("true_type_idx"), -1) for r in rows], dtype=np.int64)


def _logits(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    return np.log(np.clip(_probs(rows), 1e-12, 1.0))


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def _macro_f1_for_bias(logits: np.ndarray, labels: np.ndarray, bias: np.ndarray) -> float:
    valid = labels >= 0
    if not np.any(valid):
        return 0.0
    pred = np.argmax(logits[valid] + bias.reshape(1, -1), axis=1)
    return float(classification_metrics(labels[valid], y_pred=pred)["type_macro_f1"])


def fit_class_bias(
    rows: Sequence[Mapping[str, object]],
    bias_min: float = -4.0,
    bias_max: float = 4.0,
    bias_step: float = 0.25,
    passes: int = 4,
) -> Tuple[np.ndarray, Dict[str, object]]:
    logits = _logits(rows)
    labels = _labels(rows)
    candidates = np.arange(float(bias_min), float(bias_max) + float(bias_step) / 2.0, float(bias_step), dtype=np.float64)
    bias = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    best = _macro_f1_for_bias(logits, labels, bias)
    trace: List[Dict[str, object]] = [{"pass": 0, "class": "__init__", "macro_f1": best, "bias": bias.tolist()}]
    for pass_idx in range(1, max(1, int(passes)) + 1):
        changed = False
        for c, class_name in enumerate(CLASS_NAMES):
            best_c = float(bias[c])
            best_c_score = best
            for value in candidates:
                trial = bias.copy()
                trial[c] = float(value)
                score = _macro_f1_for_bias(logits, labels, trial)
                if score > best_c_score + 1e-12:
                    best_c_score = score
                    best_c = float(value)
            if abs(best_c - float(bias[c])) > 1e-12:
                bias[c] = best_c
                best = best_c_score
                changed = True
            trace.append({"pass": pass_idx, "class": class_name, "macro_f1": best, "bias": bias.tolist()})
        if not changed:
            break
    # Normalize for easier reading; adding a constant to every class does not change argmax/probs.
    bias = bias - float(np.mean(bias))
    best = _macro_f1_for_bias(logits, labels, bias)
    return bias, {"validation_macro_f1": best, "search_trace": trace}


def apply_bias(rows: Sequence[Mapping[str, object]], bias: Sequence[float]) -> List[Dict[str, object]]:
    bias_arr = np.asarray(list(bias), dtype=np.float64).reshape(1, -1)
    probs = _softmax(_logits(rows) + bias_arr)
    out = []
    for row, p in zip(rows, probs):
        pred = int(np.argmax(p))
        r = dict(row)
        r["pred_type_idx"] = pred
        r["pred_type_class"] = CLASS_NAMES[pred]
        r["pred_type_conf"] = float(p[pred])
        for col, val in zip(PROB_COLUMNS, p):
            r[col] = float(val)
        out.append(r)
    return out


def apply_source_bias(
    rows: Sequence[Mapping[str, object]],
    source_biases: Mapping[str, Sequence[float]],
    fallback_bias: Sequence[float],
) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Tuple[int, Mapping[str, object]]]] = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[str(row.get("classification_source", "") or "__missing__")].append((i, row))
    out: List[Dict[str, object] | None] = [None] * len(rows)
    for source, items in grouped.items():
        bias = source_biases.get(source, fallback_bias)
        calibrated = apply_bias([row for _, row in items], bias)
        for (idx, _), row in zip(items, calibrated):
            out[idx] = row
    return [r for r in out if r is not None]


def _support_counts(rows: Sequence[Mapping[str, object]]) -> Dict[str, int]:
    counts = Counter(safe_int(r.get("true_type_idx"), -1) for r in rows)
    return {name: int(counts.get(i, 0)) for i, name in enumerate(CLASS_NAMES)}


def _prediction_counts(rows: Sequence[Mapping[str, object]]) -> Dict[str, int]:
    counts = Counter(safe_int(r.get("pred_type_idx"), -1) for r in rows)
    return {name: int(counts.get(i, 0)) for i, name in enumerate(CLASS_NAMES)}


def present_class_summary(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not rows:
        return {"support": 0}
    metrics = metrics_from_prediction_rows(rows)
    support = _support_counts(rows)
    per_class = metrics.get("type_per_class_f1", {})
    present = [name for name, count in support.items() if count > 0]
    absent = [name for name, count in support.items() if count == 0]
    pred_counts = _prediction_counts(rows)
    absent_fp = sum(pred_counts[name] for name in absent)
    return {
        "support": len(rows),
        "support_by_class": support,
        "prediction_counts": pred_counts,
        "all_class_macro_f1": metrics.get("type_macro_f1"),
        "present_class_macro_f1": float(np.mean([safe_float(per_class.get(name), 0.0) for name in present])) if present else 0.0,
        "absent_class_false_positive_rate": float(absent_fp / max(1, len(rows))),
        "per_class_f1": per_class,
    }


def source_summaries(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    groups: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("classification_source", "") or "__missing__")].append(row)
    return {source: present_class_summary(items) for source, items in sorted(groups.items())}


def run_one_model(exp_dir: Path, out_dir: Path, args: argparse.Namespace) -> Dict[str, object]:
    val_rows = read_csv_rows(exp_dir / "best_val_predictions.csv")
    test_rows = read_csv_rows(exp_dir / "test_predictions.csv")
    model_out = out_dir / exp_dir.name
    model_out.mkdir(parents=True, exist_ok=True)

    raw_val_metrics = metrics_from_prediction_rows(val_rows)
    raw_test_metrics = metrics_from_prediction_rows(test_rows)

    global_bias, global_fit = fit_class_bias(val_rows, args.bias_min, args.bias_max, args.bias_step, args.passes)
    global_val = apply_bias(val_rows, global_bias)
    global_test = apply_bias(test_rows, global_bias)
    global_val_metrics = metrics_from_prediction_rows(global_val)
    global_test_metrics = metrics_from_prediction_rows(global_test)

    val_by_source: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in val_rows:
        val_by_source[str(row.get("classification_source", "") or "__missing__")].append(row)
    source_biases: Dict[str, List[float]] = {}
    source_fit: Dict[str, object] = {}
    for source, items in sorted(val_by_source.items()):
        if len(items) < args.min_source_val_rows:
            continue
        bias, fit = fit_class_bias(items, args.bias_min, args.bias_max, args.bias_step, args.passes)
        source_biases[source] = bias.tolist()
        source_fit[source] = fit
    source_val = apply_source_bias(val_rows, source_biases, global_bias.tolist())
    source_test = apply_source_bias(test_rows, source_biases, global_bias.tolist())
    source_val_metrics = metrics_from_prediction_rows(source_val)
    source_test_metrics = metrics_from_prediction_rows(source_test)

    write_csv_rows(model_out / "test_predictions_global_bias.csv", global_test, stage2a_prediction_fields(True))
    write_csv_rows(model_out / "test_predictions_source_aware_bias.csv", source_test, stage2a_prediction_fields(True))
    save_json(
        model_out / "bias.json",
        {
            "model": exp_dir.name,
            "global_bias": {name: float(global_bias[i]) for i, name in enumerate(CLASS_NAMES)},
            "global_fit": global_fit,
            "source_biases": {
                source: {name: float(values[i]) for i, name in enumerate(CLASS_NAMES)}
                for source, values in source_biases.items()
            },
            "source_fit": source_fit,
            "guardrail": "Biases are fit on best_val_predictions.csv only. Test predictions are transformed once for reporting.",
        },
    )
    metrics_payload = {
        "model": exp_dir.name,
        "raw_val_metrics": raw_val_metrics,
        "raw_test_metrics": raw_test_metrics,
        "global_bias_val_metrics": global_val_metrics,
        "global_bias_test_metrics": global_test_metrics,
        "source_aware_bias_val_metrics": source_val_metrics,
        "source_aware_bias_test_metrics": source_test_metrics,
        "raw_test_source_summaries": source_summaries(test_rows),
        "global_bias_test_source_summaries": source_summaries(global_test),
        "source_aware_bias_test_source_summaries": source_summaries(source_test),
        "raw_prediction_counts": _prediction_counts(test_rows),
        "global_bias_prediction_counts": _prediction_counts(global_test),
        "source_aware_bias_prediction_counts": _prediction_counts(source_test),
    }
    save_json(model_out / "m11_calibration_metrics.json", metrics_payload)
    return {
        "model": exp_dir.name,
        "raw_val_macro_f1": raw_val_metrics.get("type_macro_f1"),
        "raw_test_macro_f1": raw_test_metrics.get("type_macro_f1"),
        "global_bias_val_macro_f1": global_val_metrics.get("type_macro_f1"),
        "global_bias_test_macro_f1": global_test_metrics.get("type_macro_f1"),
        "source_aware_bias_val_macro_f1": source_val_metrics.get("type_macro_f1"),
        "source_aware_bias_test_macro_f1": source_test_metrics.get("type_macro_f1"),
        "raw_test_accuracy": raw_test_metrics.get("type_accuracy"),
        "global_bias_test_accuracy": global_test_metrics.get("type_accuracy"),
        "source_aware_bias_test_accuracy": source_test_metrics.get("type_accuracy"),
        "raw_test_per_class_f1": raw_test_metrics.get("type_per_class_f1"),
        "global_bias_test_per_class_f1": global_test_metrics.get("type_per_class_f1"),
        "source_aware_bias_test_per_class_f1": source_test_metrics.get("type_per_class_f1"),
        "out_dir": str(model_out),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    exp_dirs = [
        p
        for p in sorted(args.experiment_root.glob(args.model_glob))
        if (p / "best_val_predictions.csv").exists() and (p / "test_predictions.csv").exists()
    ]
    if not exp_dirs:
        raise FileNotFoundError(f"No model prediction directories found under {args.experiment_root} matching {args.model_glob}")
    rows = [run_one_model(p, args.out_dir, args) for p in exp_dirs]
    best_global = max(rows, key=lambda r: safe_float(r.get("global_bias_val_macro_f1"), 0.0))
    best_source = max(rows, key=lambda r: safe_float(r.get("source_aware_bias_val_macro_f1"), 0.0))
    save_json(
        args.out_dir / "m11_type_bias_summary.json",
        {
            "protocol_note": "Global and source-aware biases are selected by validation macro-F1 only; source-aware is diagnostic unless classification_source is deployable.",
            "experiment_root": str(args.experiment_root),
            "models": rows,
            "best_global_by_validation_macro_f1": best_global,
            "best_source_aware_by_validation_macro_f1": best_source,
        },
    )
    print("[done] calibrated", len(rows), "models")
    print("[done] wrote", args.out_dir / "m11_type_bias_summary.json")


if __name__ == "__main__":
    main()
