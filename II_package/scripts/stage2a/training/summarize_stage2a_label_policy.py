#!/usr/bin/env python3
"""Summarize Stage-2a type experiments under merged/dropped label policies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


CLASS_NAMES = ["residential_small", "residential_multi", "commercial", "institutional", "other"]
PROB_COLUMNS = [f"prob_{name}" for name in CLASS_NAMES]
POLICY_ACTIVE_CLASSES = {
    "merge_institutional_other": ["residential_small", "residential_multi", "commercial", "other"],
    "drop_institutional": ["residential_small", "residential_multi", "commercial", "other"],
    "drop_other": ["residential_small", "residential_multi", "commercial", "institutional"],
    "drop_institutional_other": ["residential_small", "residential_multi", "commercial"],
}
POLICIES = tuple(POLICY_ACTIVE_CLASSES)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize label-policy Stage2a type experiments.")
    p.add_argument("--experiment_root", type=Path, action="append", required=True)
    p.add_argument("--model_glob", type=str, action="append", default=None)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, default=None)
    p.add_argument(
        "--protocol_note",
        type=str,
        default=(
            "M20 compares semantic label policies with a fixed split assignment carried through each manifest. "
            "Candidate ranking is validation-only; held-out test metrics are present only when the run produced a test split."
        ),
    )
    return p.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, sort_keys=True)


def to_jsonable(obj: object) -> object:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = -1) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def mean(values: Sequence[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def std(values: Sequence[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.std(vals, ddof=0)) if vals else None


def policy_from_text(text: str) -> str | None:
    for policy in sorted(POLICIES, key=len, reverse=True):
        if policy in text:
            return policy
    return None


def infer_policy(exp_dir: Path, config: Mapping[str, object]) -> str:
    labels_csv = str(config.get("labels_csv", "") or "")
    policy = policy_from_text(exp_dir.name) or policy_from_text(labels_csv)
    if policy is None:
        raise ValueError(f"Could not infer label policy for {exp_dir}; include policy name in run name or labels_csv")
    return policy


def infer_recipe(exp_dir: Path, policy: str) -> str:
    name = exp_dir.name
    prefix = f"m20_{policy}_"
    if name.startswith(prefix):
        recipe = name[len(prefix) :]
    else:
        recipe = name
    recipe = re.sub(r"_seed\d+$", "", recipe)
    return recipe


def active_raw_indices(policy: str) -> List[int]:
    active = POLICY_ACTIVE_CLASSES[policy]
    return [CLASS_NAMES.index(name) for name in active]


def policy_probabilities(row: Mapping[str, object], policy: str) -> np.ndarray:
    raw = np.asarray([safe_float(row.get(col), 0.0) for col in PROB_COLUMNS], dtype=np.float64)
    if policy == "merge_institutional_other":
        probs = np.asarray([raw[0], raw[1], raw[2], raw[3] + raw[4]], dtype=np.float64)
    else:
        probs = raw[active_raw_indices(policy)]
    total = float(probs.sum())
    if total > 0:
        probs = probs / total
    return probs


def true_policy_index(row: Mapping[str, object], policy: str) -> int:
    raw_idx = safe_int(row.get("true_type_idx"), -1)
    if policy == "merge_institutional_other" and raw_idx in (3, 4):
        raw_name = "other"
    elif 0 <= raw_idx < len(CLASS_NAMES):
        raw_name = CLASS_NAMES[raw_idx]
    else:
        raw_name = str(row.get("true_type_class", "") or "")
    active = POLICY_ACTIVE_CLASSES[policy]
    return active.index(raw_name) if raw_name in active else -1


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < n_classes and 0 <= int(p) < n_classes:
            cm[int(t), int(p)] += 1
    return cm


def per_class_stats(cm: np.ndarray, names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(names):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        out[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(cm[i, :].sum()),
            "predicted": int(cm[:, i].sum()),
        }
    return out


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    if len(confidences) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidences[mask].mean()))
    return ece


def policy_metrics(rows: Sequence[Mapping[str, object]], policy: str) -> Dict[str, object]:
    active = POLICY_ACTIVE_CLASSES[policy]
    y_true: List[int] = []
    probs: List[np.ndarray] = []
    raw_argmax: List[int] = []
    skipped = 0
    for row in rows:
        target = true_policy_index(row, policy)
        if target < 0:
            skipped += 1
            continue
        y_true.append(target)
        p = policy_probabilities(row, policy)
        probs.append(p)
        raw = np.asarray([safe_float(row.get(col), 0.0) for col in PROB_COLUMNS], dtype=np.float64)
        raw_argmax.append(int(raw.argmax()) if raw.size else -1)

    if not y_true:
        return {
            "active_class_names": active,
            "type_accuracy": 0.0,
            "type_macro_f1": 0.0,
            "type_nll": 0.0,
            "type_ece": 0.0,
            "skipped_rows": skipped,
            "support": 0,
        }

    prob_arr = np.vstack(probs)
    y = np.asarray(y_true, dtype=np.int64)
    pred = prob_arr.argmax(axis=1)
    cm = confusion_matrix(y, pred, len(active))
    stats = per_class_stats(cm, active)
    f1_values = [stats[name]["f1"] for name in active]
    correct = pred == y
    nll = -np.log(np.clip(prob_arr[np.arange(len(y)), y], 1e-12, 1.0))
    active_raw = set(active_raw_indices(policy))
    if policy == "merge_institutional_other":
        invalid_raw_argmax_rate = 0.0
    else:
        invalid_raw_argmax_rate = float(sum(1 for idx in raw_argmax if idx not in active_raw) / max(1, len(raw_argmax)))
    return {
        "active_class_names": active,
        "support": int(len(y)),
        "skipped_rows": int(skipped),
        "type_accuracy": float(correct.mean()),
        "type_macro_f1": float(np.mean(f1_values)),
        "type_nll": float(nll.mean()),
        "type_ece": expected_calibration_error(prob_arr.max(axis=1), correct.astype(np.float64)),
        "type_per_class": stats,
        "type_per_class_f1": {name: stats[name]["f1"] for name in active},
        "confusion_matrix": cm.tolist(),
        "prediction_counts": {active[i]: int(cm[:, i].sum()) for i in range(len(active))},
        "support_by_class": {active[i]: int(cm[i, :].sum()) for i in range(len(active))},
        "raw_argmax_counts": {CLASS_NAMES[k]: int(v) for k, v in sorted(Counter(raw_argmax).items()) if 0 <= k < len(CLASS_NAMES)},
        "invalid_raw_argmax_rate": invalid_raw_argmax_rate,
    }


def source_summaries(rows: Sequence[Mapping[str, object]], policy: str) -> Dict[str, object]:
    groups: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("classification_source", "") or "__missing__")].append(row)
    return {source: policy_metrics(items, policy) for source, items in sorted(groups.items())}


def experiment_dirs(roots: Sequence[Path], globs: Sequence[str] | None) -> List[Path]:
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
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                out.append(path)
    return out


def find_policy_audit(config: Mapping[str, object]) -> Dict[str, object]:
    labels_csv = Path(str(config.get("labels_csv", "") or ""))
    candidates = []
    if labels_csv:
        candidates.append(labels_csv.with_name(f"{labels_csv.stem}_policy_audit.json"))
        candidates.append(Path.cwd() / labels_csv.with_name(f"{labels_csv.stem}_policy_audit.json"))
    for path in candidates:
        if path.exists():
            return load_json(path)
    return {}


def best_validation_from_history(out_dir: Path) -> Dict[str, object]:
    path = out_dir / "metrics_history.jsonl"
    if not path.exists():
        return {}
    best: Dict[str, object] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if best is None or safe_float(row.get("selection_score"), 0.0) < safe_float(best.get("selection_score"), 0.0):
            best = row
    if best is None:
        return {}
    return {
        "val_best_epoch": best.get("epoch"),
        "val_selection_score": best.get("selection_score"),
        "val_selection_metric": best.get("selection_metric"),
    }


def candidate_key(policy: str, recipe: str, config: Mapping[str, object]) -> str:
    fields = [
        "backbone_name",
        "input_mode",
        "pooling_mode",
        "type_geometry_mode",
        "type_loss_mode",
        "logit_adjust_tau",
        "sampler_mode",
        "class_weight_mode",
        "split_col",
    ]
    extras = "|".join(f"{field}={config.get(field)}" for field in fields)
    return f"policy={policy}|recipe={recipe}|{extras}"


def summarize_experiment(exp_dir: Path) -> Dict[str, object]:
    config = load_json(exp_dir / "train_config.json")
    policy = infer_policy(exp_dir, config)
    recipe = infer_recipe(exp_dir, policy)
    val_rows = read_csv_rows(exp_dir / "best_val_predictions.csv")
    test_rows = read_csv_rows(exp_dir / "test_predictions.csv") if (exp_dir / "test_predictions.csv").exists() else []
    val_metrics = policy_metrics(val_rows, policy)
    test_metrics = policy_metrics(test_rows, policy) if test_rows else {}
    audit = find_policy_audit(config)
    row = {
        "name": exp_dir.name,
        "out_dir": str(exp_dir),
        "policy": policy,
        "recipe": recipe,
        "candidate_key": candidate_key(policy, recipe, config),
        "seed": config.get("seed"),
        "split_seed": config.get("split_seed"),
        "config": config,
        "policy_audit": audit,
        "validation_policy_metrics": val_metrics,
        "test_policy_metrics": test_metrics,
        "val_source_summaries": source_summaries(val_rows, policy),
        "test_source_summaries": source_summaries(test_rows, policy) if test_rows else {},
        "summary": load_json(exp_dir / "summary.json"),
    }
    row.update(best_validation_from_history(exp_dir))
    row["val_policy_macro_f1"] = val_metrics.get("type_macro_f1")
    row["val_policy_accuracy"] = val_metrics.get("type_accuracy")
    row["test_policy_macro_f1"] = test_metrics.get("type_macro_f1") if test_metrics else None
    row["test_policy_accuracy"] = test_metrics.get("type_accuracy") if test_metrics else None
    row["retained_fraction"] = audit.get("retained_fraction")
    row["rows_after"] = audit.get("rows_after")
    row["active_class_count"] = len(POLICY_ACTIVE_CLASSES[policy])
    return row


def summarize_groups(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("candidate_key", ""))].append(row)
    out: List[Dict[str, object]] = []
    for key, items in sorted(grouped.items()):
        vals = [safe_float(item.get("val_policy_macro_f1"), float("nan")) for item in items]
        tests = [safe_float(item.get("test_policy_macro_f1"), float("nan")) for item in items]
        example = items[0]
        out.append(
            {
                "candidate_key": key,
                "policy": example.get("policy"),
                "recipe": example.get("recipe"),
                "run_count": len(items),
                "seeds": [item.get("seed") for item in items],
                "names": [item.get("name") for item in items],
                "active_class_count": example.get("active_class_count"),
                "retained_fraction": example.get("retained_fraction"),
                "rows_after": example.get("rows_after"),
                "mean_val_policy_macro_f1": mean(vals),
                "std_val_policy_macro_f1": std(vals),
                "mean_test_policy_macro_f1_report_only": mean(tests),
                "std_test_policy_macro_f1_report_only": std(tests),
            }
        )
    out.sort(key=lambda row: safe_float(row.get("mean_val_policy_macro_f1"), -1.0), reverse=True)
    return out


def best_by_policy(groups: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    out = {}
    for policy in POLICIES:
        policy_groups = [g for g in groups if g.get("policy") == policy]
        out[policy] = policy_groups[0] if policy_groups else None
    return out


def write_csv_summary(rows: Sequence[Mapping[str, object]], groups: Sequence[Mapping[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "level",
        "policy",
        "recipe",
        "name",
        "seed",
        "active_class_count",
        "retained_fraction",
        "rows_after",
        "val_policy_macro_f1",
        "test_policy_macro_f1",
        "val_policy_accuracy",
        "test_policy_accuracy",
        "candidate_key",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "level": "run",
                    "policy": row.get("policy"),
                    "recipe": row.get("recipe"),
                    "name": row.get("name"),
                    "seed": row.get("seed"),
                    "active_class_count": row.get("active_class_count"),
                    "retained_fraction": row.get("retained_fraction"),
                    "rows_after": row.get("rows_after"),
                    "val_policy_macro_f1": row.get("val_policy_macro_f1"),
                    "test_policy_macro_f1": row.get("test_policy_macro_f1"),
                    "val_policy_accuracy": row.get("val_policy_accuracy"),
                    "test_policy_accuracy": row.get("test_policy_accuracy"),
                    "candidate_key": row.get("candidate_key"),
                }
            )
        for group in groups:
            writer.writerow(
                {
                    "level": "group",
                    "policy": group.get("policy"),
                    "recipe": group.get("recipe"),
                    "active_class_count": group.get("active_class_count"),
                    "retained_fraction": group.get("retained_fraction"),
                    "rows_after": group.get("rows_after"),
                    "val_policy_macro_f1": group.get("mean_val_policy_macro_f1"),
                    "test_policy_macro_f1": group.get("mean_test_policy_macro_f1_report_only"),
                    "candidate_key": group.get("candidate_key"),
                }
            )


def main() -> None:
    args = parse_args()
    dirs = experiment_dirs(args.experiment_root, args.model_glob)
    if not dirs:
        roots = ", ".join(str(root) for root in args.experiment_root)
        globs = ", ".join(args.model_glob or ["*"])
        raise FileNotFoundError(f"No experiment dirs with best_val_predictions.csv under {roots} matching {globs}")
    experiments = [summarize_experiment(path) for path in dirs]
    groups = summarize_groups(experiments)
    payload = {
        "protocol_note": args.protocol_note,
        "policies": list(POLICIES),
        "experiment_roots": [str(path) for path in args.experiment_root],
        "model_globs": args.model_glob or ["*"],
        "experiments": experiments,
        "candidate_groups": groups,
        "best_by_mean_validation_policy_macro_f1": groups[0] if groups else None,
        "best_by_policy": best_by_policy(groups),
    }
    save_json(args.out_json, payload)
    if args.out_csv:
        write_csv_summary(experiments, groups, args.out_csv)
    print("[done] summarized", len(experiments), "experiments")
    print("[done] wrote", args.out_json)


if __name__ == "__main__":
    main()
