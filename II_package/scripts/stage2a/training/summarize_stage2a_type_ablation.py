#!/usr/bin/env python3
"""Summarize drop-both Stage-2a type ablations and common error modes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

try:
    from summarize_stage2a_label_policy import (
        POLICY_ACTIVE_CLASSES,
        PROB_COLUMNS,
        policy_metrics,
        policy_probabilities,
        read_csv_rows,
        safe_float,
        save_json,
        true_policy_index,
    )
except ImportError:  # pragma: no cover
    from II_package.scripts.stage2a.training.summarize_stage2a_label_policy import (
        POLICY_ACTIVE_CLASSES,
        PROB_COLUMNS,
        policy_metrics,
        policy_probabilities,
        read_csv_rows,
        safe_float,
        save_json,
        true_policy_index,
    )


DEFAULT_ERROR_PAIRS = (
    "residential_multi>residential_small",
    "residential_small>residential_multi",
    "commercial>residential_multi",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize Stage2a drop-both type ablation predictions.")
    p.add_argument("--experiment_root", type=Path, action="append", default=[])
    p.add_argument("--model_glob", type=str, action="append", default=None)
    p.add_argument(
        "--prediction_entry",
        type=str,
        action="append",
        default=[],
        help="Optional NAME=CSV entry, useful for ensembles that are not stored as train run directories.",
    )
    p.add_argument("--policy", type=str, default="drop_institutional_other", choices=sorted(POLICY_ACTIVE_CLASSES))
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--out_md", type=Path, required=True)
    p.add_argument("--examples_csv", type=Path, default=None)
    p.add_argument("--examples_per_pair", type=int, default=12)
    p.add_argument("--error_pair", type=str, action="append", default=list(DEFAULT_ERROR_PAIRS))
    return p.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def parse_prediction_entries(entries: Iterable[str]) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for raw in entries:
        if "=" not in raw:
            raise ValueError(f"--prediction_entry must be NAME=CSV, got {raw!r}")
        name, path = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"--prediction_entry has empty name: {raw!r}")
        out.append((name, Path(path.strip())))
    return out


def variant_from_name(name: str) -> str:
    if name.startswith("m25_drop_institutional_other_convnext_ce_weighted_maskm_strict_seed"):
        return "current_strict_seed" + name.rsplit("seed", 1)[-1]
    if name.startswith("m31_native3_drop_both_convnext_ce_weighted_maskm_seed"):
        return "native3_current_strict_seed" + name.rsplit("seed", 1)[-1]
    for prefix in (
        "m29_drop_institutional_other_type_ablate_",
        "m34_native3_drop_both_type_ablate_",
    ):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def prediction_records(args: argparse.Namespace) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for exp_dir in experiment_dirs(args.experiment_root, args.model_glob):
        records.append(
            {
                "name": exp_dir.name,
                "variant": variant_from_name(exp_dir.name),
                "prediction_csv": exp_dir / "best_val_predictions.csv",
                "config": load_json(exp_dir / "train_config.json"),
                "source": "experiment_dir",
            }
        )
    for name, path in parse_prediction_entries(args.prediction_entry):
        records.append(
            {
                "name": name,
                "variant": name,
                "prediction_csv": path,
                "config": {},
                "source": "prediction_entry",
            }
        )
    if not records:
        raise FileNotFoundError("No prediction records found.")
    return records


def projected_predictions(rows: Sequence[Mapping[str, object]], policy: str) -> Tuple[List[int], List[int], np.ndarray]:
    y_true: List[int] = []
    y_pred: List[int] = []
    probs: List[np.ndarray] = []
    for row in rows:
        target = true_policy_index(row, policy)
        if target < 0:
            continue
        p = policy_probabilities(row, policy)
        y_true.append(target)
        y_pred.append(int(p.argmax()))
        probs.append(p)
    if not probs:
        return y_true, y_pred, np.zeros((0, len(POLICY_ACTIVE_CLASSES[policy])), dtype=np.float64)
    return y_true, y_pred, np.vstack(probs)


def pair_key(true_name: str, pred_name: str) -> str:
    return f"{true_name}>{pred_name}"


def pair_counts(y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for t, p in zip(y_true, y_pred):
        if t == p:
            continue
        if 0 <= t < len(class_names) and 0 <= p < len(class_names):
            counts[pair_key(class_names[t], class_names[p])] += 1
    return {key: int(value) for key, value in counts.most_common()}


def collect_error_examples(
    rows: Sequence[Mapping[str, object]],
    policy: str,
    wanted_pairs: Sequence[str],
    examples_per_pair: int,
    run_name: str,
    variant: str,
) -> List[MutableMapping[str, object]]:
    class_names = POLICY_ACTIVE_CLASSES[policy]
    wanted = set(wanted_pairs)
    grouped: Dict[str, List[MutableMapping[str, object]]] = {key: [] for key in wanted_pairs}
    for row in rows:
        target = true_policy_index(row, policy)
        if target < 0:
            continue
        probs = policy_probabilities(row, policy)
        pred = int(probs.argmax())
        if pred == target:
            continue
        key = pair_key(class_names[target], class_names[pred])
        if key not in wanted:
            continue
        out: MutableMapping[str, object] = {
            "run": run_name,
            "variant": variant,
            "error_pair": key,
            "building_uid": row.get("building_uid", ""),
            "tile_base": row.get("tile_base", ""),
            "GEOID": row.get("GEOID", ""),
            "classification_source": row.get("classification_source", ""),
            "true_type_class": class_names[target],
            "pred_type_class_policy": class_names[pred],
            "pred_type_conf_policy": float(probs[pred]),
            "true_population": row.get("true_population", ""),
            "crop_path": row.get("crop_path", ""),
            "mask_path": row.get("mask_path", ""),
        }
        for i, name in enumerate(class_names):
            out[f"policy_prob_{name}"] = float(probs[i])
        grouped[key].append(out)
    selected: List[MutableMapping[str, object]] = []
    for key in wanted_pairs:
        examples = sorted(grouped.get(key, []), key=lambda r: safe_float(r.get("pred_type_conf_policy"), 0.0), reverse=True)
        selected.extend(examples[: max(0, examples_per_pair)])
    return selected


def scalar_summary(record: Mapping[str, object], rows: Sequence[Mapping[str, object]], policy: str) -> Dict[str, object]:
    metrics = policy_metrics(rows, policy)
    class_names = list(metrics.get("active_class_names", POLICY_ACTIVE_CLASSES[policy]))
    per_class = metrics.get("type_per_class", {})
    y_true, y_pred, _ = projected_predictions(rows, policy)
    summary: Dict[str, object] = {
        "name": record["name"],
        "variant": record["variant"],
        "source": record["source"],
        "prediction_csv": str(record["prediction_csv"]),
        "support": metrics.get("support"),
        "type_macro_f1": metrics.get("type_macro_f1"),
        "type_accuracy": metrics.get("type_accuracy"),
        "type_ece": metrics.get("type_ece"),
        "type_nll": metrics.get("type_nll"),
        "confusion_matrix": metrics.get("confusion_matrix"),
        "pair_counts": pair_counts(y_true, y_pred, class_names),
        "config": record.get("config", {}),
    }
    for name in class_names:
        cls = dict(per_class.get(name, {}))
        summary[f"precision_{name}"] = cls.get("precision")
        summary[f"recall_{name}"] = cls.get("recall")
        summary[f"f1_{name}"] = cls.get("f1")
        summary[f"support_{name}"] = cls.get("support")
        summary[f"predicted_{name}"] = cls.get("predicted")
    return summary


def write_csv_summary(path: Path, rows: Sequence[Mapping[str, object]], class_names: Sequence[str]) -> None:
    fields = [
        "name",
        "variant",
        "source",
        "support",
        "type_macro_f1",
        "type_accuracy",
        "type_ece",
        "type_nll",
    ]
    for name in class_names:
        fields.extend([f"precision_{name}", f"recall_{name}", f"f1_{name}", f"support_{name}", f"predicted_{name}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_examples_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    seen = set()
    preferred = [
        "run",
        "variant",
        "error_pair",
        "building_uid",
        "tile_base",
        "GEOID",
        "classification_source",
        "true_type_class",
        "pred_type_class_policy",
        "pred_type_conf_policy",
        "true_population",
        "crop_path",
        "mask_path",
    ]
    for name in preferred:
        if any(name in row for row in rows):
            fields.append(name)
            seen.add(name)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 4) -> str:
    try:
        x = float(value)
        if math.isfinite(x):
            return f"{x:.{digits}f}"
    except (TypeError, ValueError):
        pass
    return ""


def write_markdown(path: Path, rows: Sequence[Mapping[str, object]], policy: str) -> None:
    class_names = POLICY_ACTIVE_CLASSES[policy]
    lines = [
        "# Stage2a M29 Type Ablation And Error Analysis",
        "",
        f"Policy: `{policy}`",
        "",
        "## Run Metrics",
        "",
        "| Variant | Macro-F1 | Accuracy | ECE | NLL | Multi recall | Small recall | Commercial recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda r: float(r.get("type_macro_f1") or -1.0), reverse=True):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('variant')}`",
                    fmt(row.get("type_macro_f1")),
                    fmt(row.get("type_accuracy")),
                    fmt(row.get("type_ece")),
                    fmt(row.get("type_nll")),
                    fmt(row.get("recall_residential_multi")),
                    fmt(row.get("recall_residential_small")),
                    fmt(row.get("recall_commercial")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Per-Class Precision / Recall / F1", ""])
    for row in sorted(rows, key=lambda r: str(r.get("variant"))):
        lines.append(f"### `{row.get('variant')}`")
        lines.append("")
        lines.append("| Class | Precision | Recall | F1 | Support | Predicted |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for name in class_names:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{name}`",
                        fmt(row.get(f"precision_{name}")),
                        fmt(row.get(f"recall_{name}")),
                        fmt(row.get(f"f1_{name}")),
                        str(row.get(f"support_{name}") or ""),
                        str(row.get(f"predicted_{name}") or ""),
                    ]
                )
                + " |"
            )
        lines.append("")
        lines.append("Confusion matrix rows=true, columns=pred:")
        lines.append("")
        lines.append("```text")
        lines.append("classes: " + ", ".join(class_names))
        for cm_row in row.get("confusion_matrix", []) or []:
            lines.append(" ".join(str(x) for x in cm_row))
        lines.append("```")
        lines.append("")
        pairs = row.get("pair_counts", {}) or {}
        if pairs:
            lines.append("Top error pairs: " + ", ".join(f"`{k}`={v}" for k, v in list(pairs.items())[:8]))
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    class_names = POLICY_ACTIVE_CLASSES[args.policy]
    records = prediction_records(args)
    summaries: List[Dict[str, object]] = []
    examples: List[MutableMapping[str, object]] = []
    for record in records:
        pred_csv = Path(record["prediction_csv"])
        if not pred_csv.exists():
            raise FileNotFoundError(f"Missing prediction CSV: {pred_csv}")
        rows = read_csv_rows(pred_csv)
        summary = scalar_summary(record, rows, args.policy)
        summaries.append(summary)
        examples.extend(
            collect_error_examples(
                rows,
                args.policy,
                args.error_pair,
                args.examples_per_pair,
                str(record["name"]),
                str(record["variant"]),
            )
        )
    payload = {
        "policy": args.policy,
        "active_class_names": class_names,
        "records": summaries,
        "error_pairs_requested": args.error_pair,
        "error_examples_csv": str(args.examples_csv or ""),
    }
    save_json(args.out_json, payload)
    write_csv_summary(args.out_csv, summaries, class_names)
    if args.examples_csv:
        write_examples_csv(args.examples_csv, examples)
    write_markdown(args.out_md, summaries, args.policy)
    print("[done] records=", len(summaries))
    print("[done] wrote=", args.out_json)


if __name__ == "__main__":
    main()
