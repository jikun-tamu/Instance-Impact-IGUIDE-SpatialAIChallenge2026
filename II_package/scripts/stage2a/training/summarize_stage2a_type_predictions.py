#!/usr/bin/env python3
"""Summarize Stage-2a type prediction runs with source-sliced guardrails."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

CLASS_NAMES = ["residential_small", "residential_multi", "commercial", "institutional", "other"]
PROB_COLUMNS = [f"prob_{name}" for name in CLASS_NAMES]
TARGET_MINORITIES = ("institutional", "other")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize Stage-2a type prediction CSVs and source guardrails.")
    p.add_argument("--experiment_root", type=Path, action="append", required=True)
    p.add_argument("--model_glob", type=str, action="append", default=None)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, default=None)
    p.add_argument("--target_classes", type=str, default=",".join(TARGET_MINORITIES))
    p.add_argument(
        "--protocol_note",
        type=str,
        default="Candidate ranking is validation-only. Test metrics are fixed-run reports.",
    )
    return p.parse_args()


def _mean(values: Sequence[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _std(values: Sequence[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.std(vals, ddof=0)) if vals else None


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def _to_jsonable(obj: object) -> object:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, indent=2, sort_keys=True)


def _confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < n_classes and 0 <= int(p) < n_classes:
            cm[int(t), int(p)] += 1
    return cm


def _macro_f1_from_cm(cm: np.ndarray) -> tuple[float, List[float]]:
    f1s: List[float] = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        den = 2 * tp + fp + fn
        f1s.append(float(2 * tp / den) if den > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0, f1s


def metrics_from_prediction_rows(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    valid = [r for r in rows if safe_int(r.get("true_type_idx"), -1) >= 0]
    if not valid:
        return {
            "type_accuracy": 0.0,
            "type_macro_f1": 0.0,
            "type_per_class_f1": {name: 0.0 for name in CLASS_NAMES},
            "confusion_matrix": [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES],
        }
    y_true = [safe_int(r.get("true_type_idx"), -1) for r in valid]
    if all(str(r.get(col, "")) != "" for r in valid for col in PROB_COLUMNS):
        probs = np.asarray([[safe_float(r.get(col), 0.0) for col in PROB_COLUMNS] for r in valid], dtype=np.float64)
        y_pred = probs.argmax(axis=1).astype(np.int64).tolist()
    else:
        y_pred = [safe_int(r.get("pred_type_idx"), -1) for r in valid]
    cm = _confusion_matrix(y_true, y_pred, len(CLASS_NAMES))
    macro_f1, per_class = _macro_f1_from_cm(cm)
    acc = float(np.mean(np.asarray(y_true, dtype=np.int64) == np.asarray(y_pred, dtype=np.int64))) if y_true else 0.0
    return {
        "type_accuracy": acc,
        "type_macro_f1": macro_f1,
        "type_per_class_f1": {name: per_class[i] for i, name in enumerate(CLASS_NAMES)},
        "confusion_matrix": cm.tolist(),
    }


def _support_counts(rows: Sequence[Mapping[str, object]]) -> Dict[str, int]:
    counts = Counter(safe_int(r.get("true_type_idx"), -1) for r in rows)
    return {name: int(counts.get(i, 0)) for i, name in enumerate(CLASS_NAMES)}


def _prediction_counts(rows: Sequence[Mapping[str, object]]) -> Dict[str, int]:
    counts = Counter(safe_int(r.get("pred_type_idx"), -1) for r in rows)
    return {name: int(counts.get(i, 0)) for i, name in enumerate(CLASS_NAMES)}


def _prediction_false_positive_rate(rows: Sequence[Mapping[str, object]], class_name: str) -> float:
    try:
        class_idx = CLASS_NAMES.index(class_name)
    except ValueError:
        return 0.0
    non_target = [r for r in rows if safe_int(r.get("true_type_idx"), -1) != class_idx]
    if not non_target:
        return 0.0
    false_pos = sum(1 for r in non_target if safe_int(r.get("pred_type_idx"), -1) == class_idx)
    return float(false_pos / len(non_target))


def source_slice_summary(rows: Sequence[Mapping[str, object]], target_classes: Sequence[str] = TARGET_MINORITIES) -> Dict[str, object]:
    if not rows:
        return {"support": 0}
    metrics = metrics_from_prediction_rows(rows)
    support = _support_counts(rows)
    pred_counts = _prediction_counts(rows)
    per_class = metrics.get("type_per_class_f1", {})
    present = [name for name, count in support.items() if count > 0]
    absent = [name for name, count in support.items() if count == 0]
    absent_fp = sum(pred_counts[name] for name in absent)
    target_absent_fp: Dict[str, float | None] = {}
    target_fp: Dict[str, float] = {}
    for class_name in target_classes:
        class_pred_count = pred_counts.get(class_name, 0)
        target_absent_fp[class_name] = float(class_pred_count / max(1, len(rows))) if support.get(class_name, 0) == 0 else None
        target_fp[class_name] = _prediction_false_positive_rate(rows, class_name)
    return {
        "support": len(rows),
        "support_by_class": support,
        "prediction_counts": pred_counts,
        "all_class_macro_f1": metrics.get("type_macro_f1"),
        "present_class_macro_f1": float(np.mean([safe_float(per_class.get(name), 0.0) for name in present])) if present else 0.0,
        "absent_class_false_positive_rate": float(absent_fp / max(1, len(rows))),
        "target_absent_class_false_positive_rate": target_absent_fp,
        "target_false_positive_rate": target_fp,
        "per_class_f1": per_class,
    }


def source_summaries(rows: Sequence[Mapping[str, object]], target_classes: Sequence[str] = TARGET_MINORITIES) -> Dict[str, object]:
    groups: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("classification_source", "") or "__missing__")].append(row)
    return {source: source_slice_summary(items, target_classes) for source, items in sorted(groups.items())}


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _best_validation_from_history(out_dir: Path) -> Dict[str, object]:
    history_path = out_dir / "metrics_history.jsonl"
    if not history_path.exists():
        return {}
    best = None
    for line in history_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if best is None or float(row.get("selection_score", 0.0)) < float(best.get("selection_score", 0.0)):
            best = row
    if not best:
        return {}
    return {
        "val_best_epoch": best.get("epoch"),
        "val_selection_score": best.get("selection_score"),
        "val_type_accuracy": best.get("type_accuracy"),
        "val_type_macro_f1": best.get("type_macro_f1"),
        "val_type_per_class_f1": best.get("type_per_class_f1"),
        "val_selection_metric": best.get("selection_metric"),
    }


def _candidate_key(config: Mapping[str, object]) -> str:
    fields = [
        "backbone_name",
        "input_mode",
        "pooling_mode",
        "ring_radius_px",
        "type_geometry_mode",
        "type_loss_mode",
        "logit_adjust_tau",
        "class_balanced_beta",
        "focal_gamma",
        "sampler_mode",
        "class_weight_mode",
        "split_seed",
    ]
    return "|".join(f"{field}={config.get(field)}" for field in fields)


def _source_metric(summaries: Mapping[str, object], source: str, metric: str) -> float | None:
    row = summaries.get(source)
    if not isinstance(row, Mapping):
        return None
    value = row.get(metric)
    return safe_float(value, float("nan")) if value is not None else None


def _experiment_dirs(roots: Sequence[Path], globs: Sequence[str] | None) -> List[Path]:
    patterns = list(globs or ["*"])
    out: List[Path] = []
    seen = set()
    for root in roots:
        for pattern in patterns:
            for path in sorted(root.glob(pattern)):
                if not path.is_dir():
                    continue
                if not (path / "best_val_predictions.csv").exists():
                    continue
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                out.append(path)
    return out


def summarize_experiment(exp_dir: Path, target_classes: Sequence[str]) -> Dict[str, object]:
    config = _load_json(exp_dir / "train_config.json")
    summary = _load_json(exp_dir / "summary.json")
    test_metrics = _load_json(exp_dir / "test_metrics.json")
    val_rows = read_csv_rows(exp_dir / "best_val_predictions.csv")
    test_rows = read_csv_rows(exp_dir / "test_predictions.csv") if (exp_dir / "test_predictions.csv").exists() else []
    val_metrics = metrics_from_prediction_rows(val_rows)
    test_metrics_from_rows = metrics_from_prediction_rows(test_rows) if test_rows else {}
    val_source = source_summaries(val_rows, target_classes)
    test_source = source_summaries(test_rows, target_classes) if test_rows else {}
    row = {
        "name": exp_dir.name,
        "out_dir": str(exp_dir),
        "candidate_key": _candidate_key(config),
        "seed": config.get("seed"),
        "split_seed": config.get("split_seed"),
        "config": config,
        "summary": summary,
        "test_metrics": test_metrics or test_metrics_from_rows,
        "val_metrics_from_predictions": val_metrics,
        "test_metrics_from_predictions": test_metrics_from_rows,
        "val_source_summaries": val_source,
        "test_source_summaries": test_source,
    }
    row.update(_best_validation_from_history(exp_dir))
    row["val_type_macro_f1"] = row.get("val_type_macro_f1", val_metrics.get("type_macro_f1"))
    row["test_type_macro_f1"] = (test_metrics or test_metrics_from_rows).get("type_macro_f1")
    row["test_type_accuracy"] = (test_metrics or test_metrics_from_rows).get("type_accuracy")
    row["parcel_code_val_present_macro_f1"] = _source_metric(val_source, "parcel_code", "present_class_macro_f1")
    row["parcel_code_test_present_macro_f1"] = _source_metric(test_source, "parcel_code", "present_class_macro_f1")
    row["area_heuristic_val_present_macro_f1"] = _source_metric(val_source, "area_heuristic", "present_class_macro_f1")
    row["area_heuristic_test_present_macro_f1"] = _source_metric(test_source, "area_heuristic", "present_class_macro_f1")
    return row


def summarize_groups(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("candidate_key", ""))].append(row)
    groups: List[Dict[str, object]] = []
    for key, items in sorted(grouped.items()):
        val_scores = [safe_float(item.get("val_type_macro_f1"), float("nan")) for item in items]
        test_scores = [safe_float(item.get("test_type_macro_f1"), float("nan")) for item in items]
        parcel_val = [safe_float(item.get("parcel_code_val_present_macro_f1"), float("nan")) for item in items]
        parcel_test = [safe_float(item.get("parcel_code_test_present_macro_f1"), float("nan")) for item in items]
        example_cfg = dict(items[0].get("config") or {})
        groups.append(
            {
                "candidate_key": key,
                "run_count": len(items),
                "seeds": [item.get("seed") for item in items],
                "names": [item.get("name") for item in items],
                "config_excerpt": {
                    "backbone_name": example_cfg.get("backbone_name"),
                    "input_mode": example_cfg.get("input_mode"),
                    "pooling_mode": example_cfg.get("pooling_mode"),
                    "ring_radius_px": example_cfg.get("ring_radius_px"),
                    "type_geometry_mode": example_cfg.get("type_geometry_mode"),
                    "type_loss_mode": example_cfg.get("type_loss_mode"),
                    "logit_adjust_tau": example_cfg.get("logit_adjust_tau"),
                    "class_balanced_beta": example_cfg.get("class_balanced_beta"),
                    "focal_gamma": example_cfg.get("focal_gamma"),
                    "split_seed": example_cfg.get("split_seed"),
                },
                "mean_val_type_macro_f1": _mean(val_scores),
                "std_val_type_macro_f1": _std(val_scores),
                "mean_test_type_macro_f1_report_only": _mean(test_scores),
                "std_test_type_macro_f1_report_only": _std(test_scores),
                "mean_parcel_code_val_present_macro_f1": _mean(parcel_val),
                "mean_parcel_code_test_present_macro_f1_report_only": _mean(parcel_test),
            }
        )
    groups.sort(key=lambda row: safe_float(row.get("mean_val_type_macro_f1"), -1.0), reverse=True)
    return groups


def write_csv_summary(rows: Sequence[Mapping[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "seed",
        "split_seed",
        "val_type_macro_f1",
        "test_type_macro_f1",
        "test_type_accuracy",
        "parcel_code_val_present_macro_f1",
        "parcel_code_test_present_macro_f1",
        "area_heuristic_val_present_macro_f1",
        "area_heuristic_test_present_macro_f1",
        "candidate_key",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    args = parse_args()
    target_classes = [x.strip() for x in args.target_classes.split(",") if x.strip()]
    exp_dirs = _experiment_dirs(args.experiment_root, args.model_glob)
    if not exp_dirs:
        patterns = ", ".join(args.model_glob or ["*"])
        roots = ", ".join(str(p) for p in args.experiment_root)
        raise FileNotFoundError(f"No experiment dirs with predictions found under {roots} matching {patterns}")
    rows = [summarize_experiment(path, target_classes) for path in exp_dirs]
    groups = summarize_groups(rows)
    payload = {
        "protocol_note": args.protocol_note,
        "target_classes_for_absent_fp": target_classes,
        "experiment_roots": [str(path) for path in args.experiment_root],
        "model_globs": args.model_glob or ["*"],
        "experiments": rows,
        "candidate_groups": groups,
        "best_by_mean_validation_macro_f1": groups[0] if groups else None,
    }
    save_json(args.out_json, payload)
    if args.out_csv:
        write_csv_summary(rows, args.out_csv)
    print("[done] summarized", len(rows), "experiments")
    print("[done] wrote", args.out_json)


if __name__ == "__main__":
    main()
