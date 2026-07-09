#!/usr/bin/env python3
"""Build a unified instance-level presentation table from Stage1/2a/2b outputs.

This is a driver/reporting utility (no modeling):
- Stage1 shared artifacts: geometry/crop metadata + SAM3 confidence
- Stage2a predictions: population/type/confidence
- Stage2b predictions: damage + calibrated confidence/uncertainty metrics
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Merge Stage1 + Stage2a + Stage2b into presentation outputs.")
    p.add_argument("--shared_csv", type=Path, required=True, help="shared_instance_samples.csv")
    p.add_argument("--stage2a_csv", type=Path, required=True, help="Stage2a inference output CSV")
    p.add_argument("--stage2b_jsonl", type=Path, required=True, help="Stage2b ensemble inference JSONL")
    p.add_argument("--out_csv", type=Path, required=True, help="Merged presentation CSV")
    p.add_argument(
        "--out_summary_json",
        type=Path,
        default=None,
        help="Optional summary JSON path (terminal summary is always printed)",
    )
    p.add_argument("--out_top_uncertain_csv", type=Path, default=None, help="Optional top-uncertain CSV path")
    p.add_argument("--top_k_uncertain", type=int, default=30, help="Rows to keep in top-uncertain output")
    p.add_argument("--print_top_n", type=int, default=10, help="Print top-N uncertain rows in terminal")
    p.add_argument("--log_every", type=int, default=500, help="Progress interval")
    return p.parse_args()


def read_csv_rows(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), (reader.fieldnames or [])


def read_jsonl_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
    return rows


def _f(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return float(default)


def _i(x, default=-1):
    try:
        return int(x)
    except Exception:
        return int(default)


def quantiles(xs):
    xs = [x for x in xs if np.isfinite(x)]
    if not xs:
        return {}
    arr = np.asarray(xs, dtype=np.float64)
    qs = [0, 25, 50, 75, 90, 95, 99]
    vals = np.percentile(arr, qs)
    return {f"p{q}": float(v) for q, v in zip(qs, vals)}


def _fmt_quantiles(qdict):
    if not qdict:
        return "n/a"
    order = ["p0", "p25", "p50", "p75", "p90", "p95", "p99"]
    parts = []
    for k in order:
        if k in qdict:
            parts.append(f"{k}={qdict[k]:.4f}")
    return ", ".join(parts)


def main():
    args = parse_args()
    for p in [args.shared_csv, args.stage2a_csv, args.stage2b_jsonl]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.out_summary_json is not None:
        args.out_summary_json.parent.mkdir(parents=True, exist_ok=True)
    if args.out_top_uncertain_csv is not None:
        args.out_top_uncertain_csv.parent.mkdir(parents=True, exist_ok=True)

    shared_rows, _ = read_csv_rows(args.shared_csv)
    s2a_rows, _ = read_csv_rows(args.stage2a_csv)
    s2b_rows = read_jsonl_rows(args.stage2b_jsonl)

    # Canonical key is building instance uid.
    shared_by_uid = {r.get("bldg_uid", ""): r for r in shared_rows if r.get("bldg_uid", "")}
    s2a_by_uid = {r.get("building_uid", ""): r for r in s2a_rows if r.get("building_uid", "")}
    s2b_by_uid = {r.get("bldg_uid", ""): r for r in s2b_rows if r.get("bldg_uid", "")}

    all_uids = sorted(set(shared_by_uid.keys()) | set(s2a_by_uid.keys()) | set(s2b_by_uid.keys()))
    merged = []
    for idx, uid in enumerate(all_uids, start=1):
        r1 = shared_by_uid.get(uid, {})
        r2 = s2a_by_uid.get(uid, {})
        r3 = s2b_by_uid.get(uid, {})

        out = {
            "instance_id": uid,
            "tile_id": r1.get("tile_id", "") or r2.get("tile_id", "") or r3.get("tile_id", ""),
            "event_id": r1.get("event_id", "") or r2.get("event_id", "") or r3.get("event_id", ""),
            "hazard_type": r1.get("hazard_type", "") or r2.get("hazard_type", ""),
            # Stage1 confidence
            "stage1_sam3_confidence": _f(r1.get("sam3_confidence", np.nan)),
            # Stage2a outputs/confidence
            "stage2a_pred_population": _f(r2.get("pred_population", np.nan)),
            "stage2a_pred_log1p_population": _f(r2.get("pred_log1p_population", np.nan)),
            "stage2a_pred_type_idx": _i(r2.get("pred_type_idx", -1)),
            "stage2a_pred_type_class": r2.get("pred_type_class", ""),
            "stage2a_pred_type_conf": _f(r2.get("pred_type_conf", np.nan)),
            # Stage2b outputs/confidence/uncertainty
            "stage2b_pred_damage_class": (_i(r3.get("y_pred_ensemble", -1)) if r3 else ""),
            "stage2b_expected_severity": _f(r3.get("expected_severity_ensemble", np.nan)),
            "stage2b_pmax": _f(r3.get("pmax", np.nan)),
            "stage2b_margin": _f(r3.get("margin", np.nan)),
            "stage2b_entropy": _f(r3.get("entropy", np.nan)),
            "stage2b_var_predicted_class_prob_weighted": _f(
                r3.get("var_predicted_class_prob_weighted", np.nan)
            ),
            "stage2b_var_expected_severity_weighted": _f(r3.get("var_expected_severity_weighted", np.nan)),
            "stage2b_calibration_method": r3.get("calibration_method", ""),
            # Pointers for visualization/debug
            "pre_crop": r1.get("pre_crop", ""),
            "post_crop": r1.get("post_crop", ""),
            "mask_M": r1.get("mask_M", ""),
            "mask_R": r1.get("mask_R", ""),
            "m_area_px": _f(r1.get("m_area_px", np.nan)),
            "r_area_px": _f(r1.get("r_area_px", np.nan)),
            # Join flags
            "has_stage1": int(bool(r1)),
            "has_stage2a": int(bool(r2)),
            "has_stage2b": int(bool(r3)),
        }

        # Optional simple driver score for ranking review, not a trained model output.
        # Higher means potentially higher human-impact + damage.
        pop = out["stage2a_pred_population"]
        sev = out["stage2b_expected_severity"]
        out["driver_exposure_damage_score"] = float(pop * sev) if np.isfinite(pop) and np.isfinite(sev) else np.nan

        merged.append(out)
        if idx % max(1, args.log_every) == 0:
            print(f"[present_instance_results] merged={idx}")

    fields = [
        "instance_id",
        "tile_id",
        "event_id",
        "hazard_type",
        "stage1_sam3_confidence",
        "stage2a_pred_population",
        "stage2a_pred_log1p_population",
        "stage2a_pred_type_idx",
        "stage2a_pred_type_class",
        "stage2a_pred_type_conf",
        "stage2b_pred_damage_class",
        "stage2b_expected_severity",
        "stage2b_pmax",
        "stage2b_margin",
        "stage2b_entropy",
        "stage2b_var_predicted_class_prob_weighted",
        "stage2b_var_expected_severity_weighted",
        "stage2b_calibration_method",
        "driver_exposure_damage_score",
        "m_area_px",
        "r_area_px",
        "pre_crop",
        "post_crop",
        "mask_M",
        "mask_R",
        "has_stage1",
        "has_stage2a",
        "has_stage2b",
    ]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(merged)

    # Summary for quick reporting.
    by_type = {}
    by_damage_all = {}
    by_damage_valid = {}
    for r in merged:
        t = r["stage2a_pred_type_class"] or "unknown"
        by_type[t] = by_type.get(t, 0) + 1
        d = str(r["stage2b_pred_damage_class"])
        by_damage_all[d] = by_damage_all.get(d, 0) + 1
        if r["has_stage2b"]:
            by_damage_valid[d] = by_damage_valid.get(d, 0) + 1

    summary = {
        "n_instances": len(merged),
        "join_coverage": {
            "has_stage1": int(sum(r["has_stage1"] for r in merged)),
            "has_stage2a": int(sum(r["has_stage2a"] for r in merged)),
            "has_stage2b": int(sum(r["has_stage2b"] for r in merged)),
            "all_three": int(sum((r["has_stage1"] and r["has_stage2a"] and r["has_stage2b"]) for r in merged)),
        },
        "stage1_sam3_confidence_quantiles": quantiles([r["stage1_sam3_confidence"] for r in merged]),
        "stage2a_pred_population_quantiles": quantiles([r["stage2a_pred_population"] for r in merged]),
        "stage2a_pred_type_conf_quantiles": quantiles([r["stage2a_pred_type_conf"] for r in merged]),
        "stage2b_pmax_quantiles": quantiles([r["stage2b_pmax"] for r in merged]),
        "stage2b_margin_quantiles": quantiles([r["stage2b_margin"] for r in merged]),
        "stage2b_entropy_quantiles": quantiles([r["stage2b_entropy"] for r in merged]),
        "stage2b_var_expected_severity_quantiles": quantiles(
            [r["stage2b_var_expected_severity_weighted"] for r in merged]
        ),
        "driver_exposure_damage_score_quantiles": quantiles([r["driver_exposure_damage_score"] for r in merged]),
        "counts_by_stage2a_type": by_type,
        "counts_by_stage2b_damage_class_all_rows": by_damage_all,
        "counts_by_stage2b_damage_class_valid_only": by_damage_valid,
        "calibration_method_values": sorted(
            list({r["stage2b_calibration_method"] for r in merged if r["stage2b_calibration_method"]})
        ),
    }
    if args.out_summary_json is not None:
        args.out_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    valid_for_uncertainty = [
        r
        for r in merged
        if r["has_stage2b"]
        and np.isfinite(_f(r["stage2b_entropy"]))
        and np.isfinite(_f(r["stage2b_pmax"]))
        and np.isfinite(_f(r["stage2b_var_expected_severity_weighted"]))
    ]
    if args.out_top_uncertain_csv is not None:
        # Rank uncertain cases with high entropy and low pmax, then high severity variance.
        ranked = list(valid_for_uncertainty)
        ranked.sort(
            key=lambda r: (
                -(_f(r["stage2b_entropy"], -1e9)),
                _f(r["stage2b_pmax"], 1e9),
                -(_f(r["stage2b_var_expected_severity_weighted"], -1e9)),
            )
        )
        top = ranked[: max(0, args.top_k_uncertain)]
        with args.out_top_uncertain_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(top)
        print("[done] wrote top-uncertain:", args.out_top_uncertain_csv, "rows=", len(top))
    else:
        ranked = list(valid_for_uncertainty)
        ranked.sort(
            key=lambda r: (
                -(_f(r["stage2b_entropy"], -1e9)),
                _f(r["stage2b_pmax"], 1e9),
                -(_f(r["stage2b_var_expected_severity_weighted"], -1e9)),
            )
        )
        top = ranked[: max(0, args.top_k_uncertain)]

    print("[done] wrote merged csv:", args.out_csv, "rows=", len(merged))
    if args.out_summary_json is not None:
        print("[done] wrote summary json:", args.out_summary_json)
    print("[summary] all_three_joined=", summary["join_coverage"]["all_three"])
    print("[summary] counts_by_stage2a_type=", summary["counts_by_stage2a_type"])
    print("[summary] counts_by_stage2b_damage_class_all_rows=", summary["counts_by_stage2b_damage_class_all_rows"])
    print(
        "[summary] counts_by_stage2b_damage_class_valid_only=",
        summary["counts_by_stage2b_damage_class_valid_only"],
    )
    print(
        "[summary] stage2b_missing_rows=",
        int(summary["join_coverage"]["has_stage1"]) - int(summary["join_coverage"]["has_stage2b"]),
    )
    print("[summary] stage1_sam3_confidence:", _fmt_quantiles(summary["stage1_sam3_confidence_quantiles"]))
    print("[summary] stage2a_pred_population:", _fmt_quantiles(summary["stage2a_pred_population_quantiles"]))
    print("[summary] stage2a_pred_type_conf:", _fmt_quantiles(summary["stage2a_pred_type_conf_quantiles"]))
    print("[summary] stage2b_pmax:", _fmt_quantiles(summary["stage2b_pmax_quantiles"]))
    print("[summary] stage2b_margin:", _fmt_quantiles(summary["stage2b_margin_quantiles"]))
    print("[summary] stage2b_entropy:", _fmt_quantiles(summary["stage2b_entropy_quantiles"]))
    print(
        "[summary] stage2b_var_expected_severity:",
        _fmt_quantiles(summary["stage2b_var_expected_severity_quantiles"]),
    )
    print("[summary] driver_exposure_damage_score:", _fmt_quantiles(summary["driver_exposure_damage_score_quantiles"]))

    n_show = min(max(0, args.print_top_n), len(top))
    if n_show > 0:
        print(f"[top_uncertain] first_{n_show} (sorted by high entropy, low pmax, high var_expected_severity):")
        for r in top[:n_show]:
            print(
                "  ",
                r["instance_id"],
                "tile=",
                r["tile_id"],
                "damage=",
                r["stage2b_pred_damage_class"],
                "pmax=",
                f"{_f(r['stage2b_pmax']):.4f}",
                "entropy=",
                f"{_f(r['stage2b_entropy']):.4f}",
                "varEy=",
                f"{_f(r['stage2b_var_expected_severity_weighted']):.6f}",
                "s1_conf=",
                f"{_f(r['stage1_sam3_confidence']):.4f}",
                "pop=",
                f"{_f(r['stage2a_pred_population']):.2f}",
                "type=",
                r["stage2a_pred_type_class"],
                "s2a_conf=",
                f"{_f(r['stage2a_pred_type_conf']):.4f}",
            )


if __name__ == "__main__":
    main()
