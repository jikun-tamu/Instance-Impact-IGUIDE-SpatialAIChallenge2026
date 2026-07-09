#!/usr/bin/env python3
"""Frozen validation-only M15 ceiling check for Stage-2a type predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

CLASS_NAMES = ["residential_small", "residential_multi", "commercial", "institutional", "other"]
PROB_COLUMNS = [f"prob_{name}" for name in CLASS_NAMES]
TARGET_CLASSES = ["institutional", "other"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a pre-registered M15 validation-only ensemble/bias ceiling check.")
    p.add_argument("--seed_pair", action="append", required=True, help="seed,generalist_dir,cascade_dir")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--alpha_values", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    p.add_argument("--bias_modes", default="none,global,source")
    p.add_argument("--bias_min", type=float, default=-2.0)
    p.add_argument("--bias_max", type=float, default=2.0)
    p.add_argument("--bias_step", type=float, default=0.5)
    p.add_argument("--bias_passes", type=int, default=3)
    p.add_argument("--min_source_val_rows", type=int, default=20)
    p.add_argument("--max_val_accuracy_drop", type=float, default=0.03)
    p.add_argument("--max_area_absent_fp_increase", type=float, default=0.01)
    return p.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_seed_pair(text: str) -> Tuple[int, Path, Path]:
    parts = [x.strip() for x in text.split(",", 2)]
    if len(parts) != 3:
        raise ValueError(f"seed_pair must be seed,generalist_dir,cascade_dir; got {text!r}")
    return int(parts[0]), Path(parts[1]), Path(parts[2])


def row_key(row: Mapping[str, object]) -> str:
    return str(row.get("building_uid", ""))


def source(row: Mapping[str, object]) -> str:
    return str(row.get("classification_source", "") or "__missing__")


def probs_from_row(row: Mapping[str, object]) -> List[float]:
    probs = [safe_float(row.get(col), 0.0) for col in PROB_COLUMNS]
    total = sum(probs)
    if total > 0:
        return [p / total for p in probs]
    pred = safe_int(row.get("pred_type_idx"), -1)
    return [1.0 if i == pred else 0.0 for i in range(len(CLASS_NAMES))]


def argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i])


def mix_probabilities(generalist: Sequence[float], cascade: Sequence[float], alpha: float) -> List[float]:
    vals = [(1.0 - alpha) * float(g) + alpha * float(c) for g, c in zip(generalist, cascade)]
    total = sum(vals)
    return [v / total for v in vals] if total > 0 else [1.0 / len(vals) for _ in vals]


def logits_from_probs(probs: Sequence[float]) -> List[float]:
    return [math.log(max(1e-12, min(1.0, float(p)))) for p in probs]


def softmax(logits: Sequence[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


def apply_probs(rows: Sequence[Mapping[str, object]], probs_by_uid: Mapping[str, Sequence[float]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        uid = row_key(row)
        probs = list(probs_by_uid[uid])
        pred = argmax(probs)
        r = dict(row)
        for col, value in zip(PROB_COLUMNS, probs):
            r[col] = float(value)
        r["pred_type_idx"] = pred
        r["pred_type_class"] = CLASS_NAMES[pred]
        r["pred_type_conf"] = float(probs[pred])
        out.append(r)
    return out


def apply_bias_to_probs(
    rows: Sequence[Mapping[str, object]],
    probs_by_uid: Mapping[str, Sequence[float]],
    fallback_bias: Sequence[float],
    source_biases: Mapping[str, Sequence[float]] | None = None,
) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    source_biases = source_biases or {}
    for row in rows:
        uid = row_key(row)
        bias = source_biases.get(source(row), fallback_bias)
        logits = logits_from_probs(probs_by_uid[uid])
        out[uid] = softmax([x + float(b) for x, b in zip(logits, bias)])
    return out


def confusion_matrix(rows: Sequence[Mapping[str, object]]) -> List[List[int]]:
    cm = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    for row in rows:
        t = safe_int(row.get("true_type_idx"), -1)
        p = safe_int(row.get("pred_type_idx"), -1)
        if 0 <= t < len(CLASS_NAMES) and 0 <= p < len(CLASS_NAMES):
            cm[t][p] += 1
    return cm


def metrics_from_rows(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    valid = [r for r in rows if safe_int(r.get("true_type_idx"), -1) >= 0]
    cm = confusion_matrix(valid)
    f1s = []
    for c in range(len(CLASS_NAMES)):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(len(CLASS_NAMES))) - tp
        fn = sum(cm[c][r] for r in range(len(CLASS_NAMES))) - tp
        den = 2 * tp + fp + fn
        f1s.append(float(2 * tp / den) if den > 0 else 0.0)
    correct = sum(cm[i][i] for i in range(len(CLASS_NAMES)))
    total = sum(sum(row) for row in cm)
    return {
        "type_accuracy": float(correct / total) if total else 0.0,
        "type_macro_f1": float(sum(f1s) / len(f1s)) if f1s else 0.0,
        "type_per_class_f1": {name: f1s[i] for i, name in enumerate(CLASS_NAMES)},
        "confusion_matrix": cm,
    }


def source_slice(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not rows:
        return {"support": 0}
    metrics = metrics_from_rows(rows)
    support = Counter(safe_int(r.get("true_type_idx"), -1) for r in rows)
    preds = Counter(safe_int(r.get("pred_type_idx"), -1) for r in rows)
    support_by_class = {name: int(support.get(i, 0)) for i, name in enumerate(CLASS_NAMES)}
    pred_by_class = {name: int(preds.get(i, 0)) for i, name in enumerate(CLASS_NAMES)}
    present = [name for name, count in support_by_class.items() if count > 0]
    per_class = metrics["type_per_class_f1"]
    target_absent_fp: Dict[str, float | None] = {}
    for class_name in TARGET_CLASSES:
        target_absent_fp[class_name] = (
            float(pred_by_class[class_name] / max(1, len(rows))) if support_by_class[class_name] == 0 else None
        )
    return {
        "support": len(rows),
        "support_by_class": support_by_class,
        "prediction_counts": pred_by_class,
        "all_class_macro_f1": metrics["type_macro_f1"],
        "present_class_macro_f1": float(sum(per_class[name] for name in present) / len(present)) if present else 0.0,
        "target_absent_class_false_positive_rate": target_absent_fp,
        "per_class_f1": per_class,
    }


def source_summaries(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    groups: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[source(row)].append(row)
    return {name: source_slice(items) for name, items in sorted(groups.items())}


def macro_f1_for_bias(rows: Sequence[Mapping[str, object]], probs_by_uid: Mapping[str, Sequence[float]], bias: Sequence[float]) -> float:
    biased = apply_bias_to_probs(rows, probs_by_uid, bias)
    return float(metrics_from_rows(apply_probs(rows, biased))["type_macro_f1"])


def fit_bias(
    rows: Sequence[Mapping[str, object]],
    probs_by_uid: Mapping[str, Sequence[float]],
    bias_min: float,
    bias_max: float,
    bias_step: float,
    passes: int,
) -> Tuple[List[float], Dict[str, object]]:
    candidates = []
    cur = bias_min
    while cur <= bias_max + bias_step / 2.0:
        candidates.append(round(cur, 8))
        cur += bias_step
    bias = [0.0 for _ in CLASS_NAMES]
    best = macro_f1_for_bias(rows, probs_by_uid, bias)
    trace = [{"pass": 0, "class": "__init__", "macro_f1": best, "bias": list(bias)}]
    for pass_idx in range(1, max(1, passes) + 1):
        changed = False
        for c, class_name in enumerate(CLASS_NAMES):
            best_value = bias[c]
            best_score = best
            for value in candidates:
                trial = list(bias)
                trial[c] = value
                score = macro_f1_for_bias(rows, probs_by_uid, trial)
                if score > best_score + 1e-12:
                    best_score = score
                    best_value = value
            if abs(best_value - bias[c]) > 1e-12:
                bias[c] = best_value
                best = best_score
                changed = True
            trace.append({"pass": pass_idx, "class": class_name, "macro_f1": best, "bias": list(bias)})
        if not changed:
            break
    mean_bias = sum(bias) / len(bias)
    bias = [x - mean_bias for x in bias]
    best = macro_f1_for_bias(rows, probs_by_uid, bias)
    return bias, {"validation_macro_f1": best, "search_trace": trace}


def build_mixed_probs(generalist_rows: Sequence[Mapping[str, object]], cascade_rows: Sequence[Mapping[str, object]], alpha: float) -> Dict[str, List[float]]:
    cascade_by_uid = {row_key(r): r for r in cascade_rows}
    out: Dict[str, List[float]] = {}
    for row in generalist_rows:
        uid = row_key(row)
        if uid not in cascade_by_uid:
            raise KeyError(f"Missing cascade row for uid={uid}")
        out[uid] = mix_probabilities(probs_from_row(row), probs_from_row(cascade_by_uid[uid]), alpha)
    return out


def source_absent_fp(summary: Mapping[str, object], class_name: str) -> float:
    area = summary.get("area_heuristic", {})
    if not isinstance(area, Mapping):
        return 0.0
    rates = area.get("target_absent_class_false_positive_rate", {})
    if not isinstance(rates, Mapping):
        return 0.0
    value = rates.get(class_name)
    return 0.0 if value is None else float(value)


def evaluate_seed_candidate(
    seed: int,
    generalist_dir: Path,
    cascade_dir: Path,
    alpha: float,
    bias_mode: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    val_generalist = read_csv_rows(generalist_dir / "best_val_predictions.csv")
    test_generalist = read_csv_rows(generalist_dir / "test_predictions.csv")
    val_cascade = read_csv_rows(cascade_dir / "best_val_predictions.csv")
    test_cascade = read_csv_rows(cascade_dir / "test_predictions.csv")
    val_probs = build_mixed_probs(val_generalist, val_cascade, alpha)
    test_probs = build_mixed_probs(test_generalist, test_cascade, alpha)
    reference_val = apply_probs(val_cascade, {row_key(r): probs_from_row(r) for r in val_cascade})
    reference_val_metrics = metrics_from_rows(reference_val)
    reference_source = source_summaries(reference_val)

    global_bias = [0.0 for _ in CLASS_NAMES]
    source_biases: Dict[str, List[float]] = {}
    bias_fit: Dict[str, object] = {}
    if bias_mode in {"global", "source"}:
        global_bias, fit = fit_bias(val_generalist, val_probs, args.bias_min, args.bias_max, args.bias_step, args.bias_passes)
        bias_fit["global"] = fit
    if bias_mode == "source":
        by_source: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
        for row in val_generalist:
            by_source[source(row)].append(row)
        for src, items in sorted(by_source.items()):
            if len(items) < args.min_source_val_rows:
                continue
            bias, fit = fit_bias(items, val_probs, args.bias_min, args.bias_max, args.bias_step, args.bias_passes)
            source_biases[src] = bias
            bias_fit[src] = fit

    val_biased_probs = apply_bias_to_probs(val_generalist, val_probs, global_bias, source_biases)
    val_rows = apply_probs(val_generalist, val_biased_probs)
    val_metrics = metrics_from_rows(val_rows)
    val_source = source_summaries(val_rows)
    accuracy_drop = float(reference_val_metrics["type_accuracy"]) - float(val_metrics["type_accuracy"])
    area_increase = {
        name: source_absent_fp(val_source, name) - source_absent_fp(reference_source, name)
        for name in TARGET_CLASSES
    }
    guardrail_pass = (
        accuracy_drop <= float(args.max_val_accuracy_drop)
        and all(value <= float(args.max_area_absent_fp_increase) for value in area_increase.values())
    )
    return {
        "seed": seed,
        "generalist_dir": str(generalist_dir),
        "cascade_dir": str(cascade_dir),
        "alpha": alpha,
        "bias_mode": bias_mode,
        "global_bias": {name: global_bias[i] for i, name in enumerate(CLASS_NAMES)},
        "source_biases": {
            src: {name: bias[i] for i, name in enumerate(CLASS_NAMES)}
            for src, bias in source_biases.items()
        },
        "bias_fit": bias_fit,
        "val_metrics": val_metrics,
        "val_source_summaries": val_source,
        "reference_val_metrics": reference_val_metrics,
        "reference_val_source_summaries": reference_source,
        "accuracy_drop_vs_m14_reference": accuracy_drop,
        "area_absent_fp_increase_vs_m14_reference": area_increase,
        "guardrail_pass": guardrail_pass,
        "_test_generalist_rows": test_generalist,
        "_test_probs": test_probs,
        "_val_rows": val_rows,
    }


def summarize_candidate(seed_results: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    val_scores = [float(r["val_metrics"]["type_macro_f1"]) for r in seed_results]
    val_acc = [float(r["val_metrics"]["type_accuracy"]) for r in seed_results]
    return {
        "alpha": seed_results[0]["alpha"],
        "bias_mode": seed_results[0]["bias_mode"],
        "mean_val_type_macro_f1": sum(val_scores) / len(val_scores),
        "std_val_type_macro_f1": (sum((x - sum(val_scores) / len(val_scores)) ** 2 for x in val_scores) / len(val_scores)) ** 0.5,
        "mean_val_type_accuracy": sum(val_acc) / len(val_acc),
        "all_seed_guardrails_pass": all(bool(r["guardrail_pass"]) for r in seed_results),
        "seed_results": [
            {
                k: v
                for k, v in r.items()
                if not k.startswith("_")
            }
            for r in seed_results
        ],
    }


def apply_selected_to_test(selected: Mapping[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for seed_result in selected["seed_results_full"]:
        test_probs = seed_result["_test_probs"]
        test_generalist = seed_result["_test_generalist_rows"]
        global_bias = [seed_result["global_bias"][name] for name in CLASS_NAMES]
        source_biases = {
            src: [bias[name] for name in CLASS_NAMES]
            for src, bias in seed_result["source_biases"].items()
        }
        biased_probs = apply_bias_to_probs(test_generalist, test_probs, global_bias, source_biases)
        pred_rows = apply_probs(test_generalist, biased_probs)
        for row in pred_rows:
            r = dict(row)
            r["m15_seed"] = seed_result["seed"]
            r["m15_alpha"] = selected["alpha"]
            r["m15_bias_mode"] = selected["bias_mode"]
            rows.append(r)
    return rows


def prediction_fields(extra_fields: Iterable[str] = ()) -> List[str]:
    base = [
        "building_uid",
        "pred_population",
        "pred_log1p_population",
        "pred_type_idx",
        "pred_type_class",
        "pred_type_conf",
        *PROB_COLUMNS,
        "crop_path",
        "mask_path",
        "tile_base",
        "GEOID",
        "classification_source",
        "true_population",
        "true_log1p_population",
        "true_type_idx",
        "true_type_class",
    ]
    return [*base, *extra_fields]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_pairs = [parse_seed_pair(text) for text in args.seed_pair]
    alphas = parse_float_list(args.alpha_values)
    bias_modes = [x.strip() for x in args.bias_modes.split(",") if x.strip()]

    candidates = []
    for alpha in alphas:
        for bias_mode in bias_modes:
            seed_results = [
                evaluate_seed_candidate(seed, generalist, cascade, alpha, bias_mode, args)
                for seed, generalist, cascade in seed_pairs
            ]
            summary = summarize_candidate(seed_results)
            summary["seed_results_full"] = seed_results
            candidates.append(summary)

    pass_candidates = [c for c in candidates if c["all_seed_guardrails_pass"]]
    ranking_pool = pass_candidates or candidates
    ranking_pool.sort(key=lambda c: (float(c["mean_val_type_macro_f1"]), float(c["mean_val_type_accuracy"])), reverse=True)
    selected = ranking_pool[0]

    test_rows = apply_selected_to_test(selected)
    test_by_seed: Dict[int, List[Mapping[str, object]]] = defaultdict(list)
    for row in test_rows:
        test_by_seed[safe_int(row.get("m15_seed"), -1)].append(row)
    per_seed_test = []
    for seed, rows in sorted(test_by_seed.items()):
        per_seed_test.append(
            {
                "seed": seed,
                "test_metrics_report_only": metrics_from_rows(rows),
                "test_source_summaries_report_only": source_summaries(rows),
            }
        )
        write_csv_rows(args.out_dir / f"m15_selected_seed{seed}_test_predictions.csv", rows, prediction_fields(["m15_seed", "m15_alpha", "m15_bias_mode"]))

    test_scores = [float(row["test_metrics_report_only"]["type_macro_f1"]) for row in per_seed_test]
    test_acc = [float(row["test_metrics_report_only"]["type_accuracy"]) for row in per_seed_test]
    selected_public = {k: v for k, v in selected.items() if k != "seed_results_full"}
    summary = {
        "protocol_note": "M15 is a frozen validation-only ceiling check. Alpha/bias/source-bias are selected by mean validation macro-F1 across seeds with guardrails; test is applied once for fixed reporting.",
        "seed_pairs": [
            {"seed": seed, "generalist_dir": str(generalist), "cascade_dir": str(cascade)}
            for seed, generalist, cascade in seed_pairs
        ],
        "selection_rule": "highest mean validation macro-F1 among candidates passing per-seed guardrails; no test metrics used for selection",
        "candidate_count": len(candidates),
        "guardrail_passing_candidate_count": len(pass_candidates),
        "selected": selected_public,
        "test_report_only": {
            "mean_test_type_macro_f1": sum(test_scores) / len(test_scores),
            "std_test_type_macro_f1": (sum((x - sum(test_scores) / len(test_scores)) ** 2 for x in test_scores) / len(test_scores)) ** 0.5,
            "mean_test_type_accuracy": sum(test_acc) / len(test_acc),
            "per_seed": per_seed_test,
        },
        "all_candidates": [
            {k: v for k, v in c.items() if k != "seed_results_full"}
            for c in sorted(candidates, key=lambda c: float(c["mean_val_type_macro_f1"]), reverse=True)
        ],
        "reviewer_stop_rule": "If fixed-report test macro-F1 remains <0.65 or minority F1 remains <0.4, retire the 0.7+ claim.",
    }
    save_json(args.out_dir / "m15_ceiling_summary.json", summary)
    write_csv_rows(args.out_dir / "m15_selected_test_predictions.csv", test_rows, prediction_fields(["m15_seed", "m15_alpha", "m15_bias_mode"]))
    print("[done] selected alpha=", selected["alpha"], "bias_mode=", selected["bias_mode"])
    print("[done] mean val macro-F1=", selected["mean_val_type_macro_f1"])
    print("[done] mean test macro-F1 report-only=", summary["test_report_only"]["mean_test_type_macro_f1"])
    print("[done] wrote", args.out_dir / "m15_ceiling_summary.json")


if __name__ == "__main__":
    main()
