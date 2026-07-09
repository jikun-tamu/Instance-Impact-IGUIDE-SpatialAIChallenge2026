#!/usr/bin/env python3
"""Write a detailed Markdown report for the drop-both Stage2a pipeline."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


M20_REFERENCE = {
    "type_macro_f1": 0.8627585602295955,
    "type_accuracy": 0.9303995006242197,
    "commercial_f1": 0.961222540592168,
    "residential_multi_f1": 0.7864734299516909,
    "residential_small_f1": 0.8405797101449275,
    "retained_rows": "21,428 / 23,014",
    "retained_fraction": 0.9310854262622751,
}

M23_REFERENCE = {
    "population_log_mae": 0.08115111615628451,
    "population_factor2_hit_rate": 0.9606741573033708,
    "population_mae": 16.525333317849913,
    "population_r2": 0.8281045090849994,
}

M24_REFERENCE = {
    "population_log_mae": 0.07384667366836706,
    "population_factor2_hit_rate": 0.9625468164794008,
    "population_mae": 14.618334654266057,
    "population_r2": 0.8463768582200457,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate the M25/M26 drop-both stability report.")
    p.add_argument("--m25_type_summary", type=Path, required=True)
    p.add_argument("--m26_population_summary", type=Path, required=True)
    p.add_argument("--m28_type_ensemble_summary", type=Path, default=None)
    p.add_argument("--m28_population_summary", type=Path, default=None)
    p.add_argument("--out_md", type=Path, required=True)
    p.add_argument("--title", type=str, default="Stage2a Drop-Both Type-To-Population Stability Report")
    return p.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def fmt(value: object, digits: int = 4) -> str:
    val = safe_float(value)
    if not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def fmt_delta(value: object, digits: int = 4) -> str:
    val = safe_float(value)
    if not math.isfinite(val):
        return "NA"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.{digits}f}"


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    vals = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std(ddof=0))


def metric_cell(stats: Mapping[str, object], digits: int = 4) -> str:
    mean = safe_float(stats.get("mean"))
    std = safe_float(stats.get("std"))
    if not math.isfinite(mean):
        return "NA"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def type_group(m25: Mapping[str, object]) -> Mapping[str, object]:
    groups = list(m25.get("candidate_groups") or [])
    if not groups:
        return {}
    return groups[0] if isinstance(groups[0], Mapping) else {}


def type_runs(m25: Mapping[str, object]) -> List[Mapping[str, object]]:
    rows = [row for row in (m25.get("experiments") or []) if isinstance(row, Mapping)]
    rows.sort(key=lambda row: safe_float(row.get("seed"), 10**9))
    return rows


def per_class_type_stats(rows: Sequence[Mapping[str, object]]) -> Dict[str, Dict[str, float]]:
    by_class: Dict[str, List[float]] = {}
    for row in rows:
        metrics = row.get("validation_policy_metrics") or {}
        per_class = metrics.get("type_per_class_f1") if isinstance(metrics, Mapping) else {}
        if not isinstance(per_class, Mapping):
            continue
        for name, value in per_class.items():
            by_class.setdefault(str(name), []).append(safe_float(value))
    out: Dict[str, Dict[str, float]] = {}
    for name, values in sorted(by_class.items()):
        mean, std = mean_std(values)
        out[name] = {"mean": mean, "std": std}
    return out


def pop_groups(m26: Mapping[str, object]) -> List[Mapping[str, object]]:
    rows = [row for row in (m26.get("groups") or []) if isinstance(row, Mapping)]
    rows.sort(key=lambda row: safe_float((row.get("population_log_mae") or {}).get("mean"), float("inf")))
    return rows


def pop_runs(m26: Mapping[str, object]) -> List[Mapping[str, object]]:
    rows = [row for row in (m26.get("runs") or []) if isinstance(row, Mapping)]
    rows.sort(key=lambda row: (str(row.get("geometry_cols")), safe_float(row.get("seed"), 10**9)))
    return rows


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return out


def write_report(
    m25: Mapping[str, object],
    m26: Mapping[str, object],
    out_md: Path,
    title: str,
    m28_type: Mapping[str, object] | None = None,
    m28_pop: Mapping[str, object] | None = None,
) -> None:
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    tgroup = type_group(m25)
    truns = type_runs(m25)
    pclasses = per_class_type_stats(truns)
    pgroups = pop_groups(m26)
    pruns = pop_runs(m26)

    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Generated: {generated}")
    lines.append("")
    lines.append("## Decision Context")
    lines.append("")
    lines.append("This report evaluates the current Stage2b-facing Stage2a pipeline:")
    lines.append("")
    lines.append("1. Drop `institutional` and `other` rows.")
    lines.append("2. Predict type over `residential_small`, `residential_multi`, and `commercial`.")
    lines.append("3. Predict population from deployable geometry plus predicted 3-class type features.")
    lines.append("")
    lines.append("The experiment does not consider adding the two dropped classes back.")
    lines.append("")
    lines.append("## Dataset And Limitation")
    lines.append("")
    lines.append(f"- Retained rows under the drop-both policy: `{M20_REFERENCE['retained_rows']}`.")
    lines.append(f"- Retained fraction: `{fmt(M20_REFERENCE['retained_fraction'])}`.")
    lines.append("- The reported type metric is a closed-world 3-class metric.")
    lines.append("- Population metrics are measured only on rows retained after dropping `institutional` and `other`.")
    lines.append("- The pipeline should not claim support for the dropped classes under this configuration.")
    lines.append("")
    lines.append("## M25 Strict Type Stability")
    lines.append("")
    if tgroup:
        lines.extend(
            table(
                ["Run count", "Seeds", "Mean macro-F1", "Std macro-F1", "M20 single-seed macro-F1"],
                [
                    [
                        str(tgroup.get("run_count")),
                        ", ".join(str(x) for x in tgroup.get("seeds", [])),
                        fmt(tgroup.get("mean_val_policy_macro_f1")),
                        fmt(tgroup.get("std_val_policy_macro_f1")),
                        fmt(M20_REFERENCE["type_macro_f1"]),
                    ]
                ],
            )
        )
    else:
        lines.append("M25 type summary was not available.")
    lines.append("")
    if truns:
        lines.extend(
            table(
                ["Seed", "Macro-F1", "Accuracy", "ECE"],
                [
                    [
                        str(row.get("seed")),
                        fmt(row.get("val_policy_macro_f1")),
                        fmt(row.get("val_policy_accuracy")),
                        fmt((row.get("validation_policy_metrics") or {}).get("type_ece")),
                    ]
                    for row in truns
                ],
            )
        )
        lines.append("")
        lines.extend(
            table(
                ["Class", "M25 F1 mean +/- std", "M20 reference F1"],
                [
                    [
                        name,
                        f"{fmt(vals.get('mean'))} +/- {fmt(vals.get('std'))}",
                        fmt(M20_REFERENCE.get(f"{name}_f1")),
                    ]
                    for name, vals in pclasses.items()
                ],
            )
        )
    lines.append("")
    lines.append("## M26 Population Stability")
    lines.append("")
    if pgroups:
        lines.extend(
            table(
                [
                    "Geometry",
                    "Runs",
                    "Log MAE mean +/- std",
                    "Factor-2 mean +/- std",
                    "MAE mean +/- std",
                    "R2 mean +/- std",
                    "Delta vs baseline mean +/- std",
                ],
                [
                    [
                        str(group.get("geometry_cols")),
                        str(group.get("run_count")),
                        metric_cell(group.get("population_log_mae") or {}),
                        metric_cell(group.get("population_factor2_hit_rate") or {}),
                        metric_cell(group.get("population_mae") or {}, digits=2),
                        metric_cell(group.get("population_r2") or {}),
                        metric_cell(group.get("delta_log_mae_vs_baseline") or {}),
                    ]
                    for group in pgroups
                ],
            )
        )
    else:
        lines.append("M26 population summary was not available.")
    lines.append("")
    if pruns:
        lines.extend(
            table(
                ["Seed", "Geometry", "Log MAE", "Delta vs baseline", "Factor-2", "MAE", "R2"],
                [
                    [
                        str(row.get("seed")),
                        str(row.get("geometry_cols")),
                        fmt(row.get("target_population_log_mae")),
                        fmt_delta(row.get("target_delta_log_mae_vs_baseline")),
                        fmt(row.get("target_population_factor2_hit_rate")),
                        fmt(row.get("target_population_mae"), digits=2),
                        fmt(row.get("target_population_r2")),
                    ]
                    for row in pruns
                ],
            )
        )
    lines.append("")
    if m28_type or m28_pop:
        lines.append("## M28 Five-Seed Type Ensemble")
        lines.append("")
        if m28_type:
            egroup = type_group(m28_type)
            eruns = type_runs(m28_type)
            if egroup:
                lines.extend(
                    table(
                        ["Ensemble runs", "Macro-F1", "Accuracy", "Reference M25 mean macro-F1"],
                        [
                            [
                                str(egroup.get("run_count")),
                                fmt(egroup.get("mean_val_policy_macro_f1")),
                                fmt(eruns[0].get("val_policy_accuracy") if eruns else float("nan")),
                                fmt(tgroup.get("mean_val_policy_macro_f1") if tgroup else float("nan")),
                            ]
                        ],
                    )
                )
                lines.append("")
        if m28_pop:
            egroups = pop_groups(m28_pop)
            if egroups:
                lines.extend(
                    table(
                        [
                            "Geometry",
                            "Log MAE",
                            "Factor-2",
                            "MAE",
                            "R2",
                            "Delta vs baseline",
                        ],
                        [
                            [
                                str(group.get("geometry_cols")),
                                metric_cell(group.get("population_log_mae") or {}),
                                metric_cell(group.get("population_factor2_hit_rate") or {}),
                                metric_cell(group.get("population_mae") or {}, digits=2),
                                metric_cell(group.get("population_r2") or {}),
                                metric_cell(group.get("delta_log_mae_vs_baseline") or {}),
                            ]
                            for group in egroups
                        ],
                    )
                )
                lines.append("")
        lines.append(
            "M28 is a Stage2b-style probability ensemble: validation probabilities are averaged across the five "
            "M25 strict type seeds, and train-row type features are averaged across the five strict OOF seed families."
        )
        lines.append("")
    lines.append("## Baseline Comparison")
    lines.append("")
    best_group = pgroups[0] if pgroups else {}
    best_log_mae = safe_float((best_group.get("population_log_mae") or {}).get("mean")) if best_group else float("nan")
    lines.extend(
        table(
            ["Model", "Log MAE", "Factor-2", "MAE", "R2"],
            [
                [
                    "M23 selected single-seed core geometry + soft type",
                    fmt(M23_REFERENCE["population_log_mae"]),
                    fmt(M23_REFERENCE["population_factor2_hit_rate"]),
                    fmt(M23_REFERENCE["population_mae"], digits=2),
                    fmt(M23_REFERENCE["population_r2"]),
                ],
                [
                    "M24 best single-seed log-footprint + soft type",
                    fmt(M24_REFERENCE["population_log_mae"]),
                    fmt(M24_REFERENCE["population_factor2_hit_rate"]),
                    fmt(M24_REFERENCE["population_mae"], digits=2),
                    fmt(M24_REFERENCE["population_r2"]),
                ],
                [
                    "M26 best multi-seed mean",
                    fmt(best_log_mae),
                    metric_cell((best_group.get("population_factor2_hit_rate") or {}) if best_group else {}),
                    metric_cell((best_group.get("population_mae") or {}) if best_group else {}, digits=2),
                    metric_cell((best_group.get("population_r2") or {}) if best_group else {}),
                ],
            ],
        )
    )
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    if math.isfinite(best_log_mae):
        delta_m24 = best_log_mae - M24_REFERENCE["population_log_mae"]
        delta_m23 = best_log_mae - M23_REFERENCE["population_log_mae"]
        lines.append(
            f"The best M26 multi-seed mean log MAE differs from the M24 single-seed best by `{fmt_delta(delta_m24)}` "
            f"and from the M23 selected single-seed model by `{fmt_delta(delta_m23)}`."
        )
    lines.append(
        "Use the strict drop-both pipeline only as a reduced-taxonomy Stage2b path: it improves focus on the "
        "residential/commercial classes but narrows class coverage by design."
    )
    lines.append("")
    lines.append("## Evidence Files")
    lines.append("")
    lines.append(f"- M25 type summary: `{m25.get('experiment_roots', [''])[0] if m25.get('experiment_roots') else 'see input JSON'}`")
    lines.append(f"- M26 population summary target model: `{m26.get('target_model', '')}`")
    lines.append("- Queue: `refine-logs/stage2a_m25_m26_drop_both_stability_queue.sh`")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[done] wrote", out_md)


def main() -> None:
    args = parse_args()
    m28_type = load_json(args.m28_type_ensemble_summary) if args.m28_type_ensemble_summary else None
    m28_pop = load_json(args.m28_population_summary) if args.m28_population_summary else None
    write_report(
        load_json(args.m25_type_summary),
        load_json(args.m26_population_summary),
        args.out_md,
        args.title,
        m28_type=m28_type,
        m28_pop=m28_pop,
    )


if __name__ == "__main__":
    main()
