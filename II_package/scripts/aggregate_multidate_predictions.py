#!/usr/bin/env python3
"""Aggregate Stage-2b predictions across multiple post dates per building instance.

Reads per-date Stage-2b JSONL files and produces five aggregation strategies:

  M1  — prob_avg:      average calibrated probabilities across tile-quality-ok dates,
                        then argmax. NOT_IDENTIFIABLE if no tiles pass.
  M1b — coverage_avg:  like M1 but additionally requires per-building crop coverage
                        (tile_quality_ok AND crop_quality_ok). NOT_IDENTIFIABLE if
                        no date has valid coverage for this building.
  M2  — majority_vote: most common y_pred across tile-quality-ok dates.
                        Tie broken by M1 prob_avg argmax. NOT_IDENTIFIABLE if no tiles pass.
  M2b — coverage_vote: **Real-world multi-image rule.** Majority vote across dates
                        where the building is identifiable (tile_quality_ok AND
                        crop_quality_ok). NOT_IDENTIFIABLE only if the building was
                        never sufficiently captured. Tie broken by highest damage class
                        (conservative: assume worst plausible outcome when ambiguous).
  M3  — quality_avg:   like M1 but requires both tile and crop quality; falls back
                        to M1 on zero valid dates.

Also computes per-instance agreement metrics:
  - n_dates_total, n_dates_used, dates_used, dates_skipped
  - label_entropy: entropy of per-date damage labels (measures instability)
  - is_unstable: True if at least two different damage classes predicted across dates

Output
------
  aggregated_predictions.jsonl  — one JSON object per building instance
  aggregated_predictions.csv    — flat CSV version for easy review
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DAMAGE_CLASSES = [0, 1, 2, 3]
N_CLASSES = 4
NOT_IDENTIFIABLE = -1   # assigned when no valid post-disaster coverage exists


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cell_run_dir", type=Path, required=True,
                   help="Root dir for this cell under multidate_experiment/.")
    p.add_argument("--out_jsonl", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    return p.parse_args()


def load_jsonl(path: Path) -> Dict[str, dict]:
    """Return {bldg_uid: record} from a Stage-2b JSONL file."""
    records: Dict[str, dict] = {}
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            uid = obj.get("bldg_uid", "")
            if uid:
                records[uid] = obj
    return records


def load_quality(quality_json: Path) -> Tuple[bool, dict]:
    """Return (tile_ok, metrics) from a quality_metrics.json file."""
    if not quality_json.exists():
        return False, {}
    metrics = json.loads(quality_json.read_text())
    return bool(metrics.get("tile_quality_ok", False)), metrics


def load_shared_csv_quality(shared_csv: Path) -> Dict[str, bool]:
    """Return {bldg_uid: quality_ok} from a per-date shared_for_date.csv."""
    result: Dict[str, bool] = {}
    if not shared_csv.exists():
        return result
    with open(shared_csv, newline="") as f:
        for row in csv.DictReader(f):
            uid = row.get("bldg_uid", "")
            qok = row.get("quality_ok", "false").lower() == "true"
            if uid:
                result[uid] = qok
    return result


def label_entropy(labels: List[int]) -> float:
    """Shannon entropy of a list of integer labels (in nats, normalized by log(N_CLASSES))."""
    if not labels:
        return 0.0
    counts = Counter(labels)
    n = len(labels)
    h = -sum((c / n) * math.log(c / n) for c in counts.values() if c > 0)
    return round(h / math.log(N_CLASSES) if N_CLASSES > 1 else 0.0, 4)


def safe_argmax(probs: List[float]) -> int:
    return int(max(range(len(probs)), key=lambda i: probs[i]))


def avg_probs(prob_lists: List[List[float]]) -> List[float]:
    """Element-wise mean of probability vectors."""
    n = len(prob_lists)
    result = [0.0] * N_CLASSES
    for pl in prob_lists:
        for i, v in enumerate(pl):
            result[i] += v / n
    return [round(v, 6) for v in result]


def majority_vote(labels: List[int], tiebreak_probs: Optional[List[float]] = None) -> int:
    counts = Counter(labels)
    max_count = max(counts.values())
    candidates = [lbl for lbl, cnt in counts.items() if cnt == max_count]
    if len(candidates) == 1:
        return candidates[0]
    # Tie: use tiebreak_probs argmax
    if tiebreak_probs is not None:
        best = max(candidates, key=lambda lbl: tiebreak_probs[lbl])
        return best
    return min(candidates)  # deterministic fallback


def main():
    args = parse_args()
    cell_run_dir = args.cell_run_dir
    dates_dir = cell_run_dir / "dates"

    if not dates_dir.exists():
        sys.exit(f"ERROR: dates_dir not found: {dates_dir}")

    # ── Discover dates ────────────────────────────────────────────────────────
    date_entries = []  # list of {date, jsonl, quality_json, shared_csv}
    for date_dir in sorted(dates_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        date = date_dir.name
        jsonl_files = list(date_dir.glob("stage2b_*.jsonl"))
        if not jsonl_files:
            continue
        date_entries.append({
            "date": date,
            "jsonl": jsonl_files[0],
            "quality_json": date_dir / "quality_metrics.json",
            "shared_csv": date_dir / "shared_for_date.csv",
        })

    if not date_entries:
        sys.exit(f"ERROR: no stage2b_*.jsonl files found under {dates_dir}")

    print(f"Found {len(date_entries)} dates: {[e['date'] for e in date_entries]}")

    # ── Load all data ─────────────────────────────────────────────────────────
    # per_date[date] = {bldg_uid: {pred, probs_cal, quality_ok_crop}}
    per_date: Dict[str, Dict[str, dict]] = {}
    date_tile_ok: Dict[str, bool] = {}

    for entry in date_entries:
        date = entry["date"]
        tile_ok, _ = load_quality(entry["quality_json"])
        date_tile_ok[date] = tile_ok
        preds = load_jsonl(entry["jsonl"])
        crop_quality = load_shared_csv_quality(entry["shared_csv"])
        per_date[date] = {}
        for uid, rec in preds.items():
            probs_cal = rec.get("ensemble_probs_calibrated") or rec.get("ensemble_probs", [])
            per_date[date][uid] = {
                "y_pred": rec.get("y_pred_ensemble", -1),
                "probs_cal": probs_cal,
                # Default False (conservative): if a building is not in the crop-quality
                # CSV, treat it as NOT covered rather than assuming valid coverage.
                "quality_ok_crop": crop_quality.get(uid, False),
                "tile_quality_ok": tile_ok,
            }

    # ── Collect all building UIDs across all dates ────────────────────────────
    all_uids: set = set()
    for date_preds in per_date.values():
        all_uids.update(date_preds.keys())

    print(f"Total unique building UIDs: {len(all_uids)}")

    # ── Per-instance aggregation ──────────────────────────────────────────────
    results = []
    for uid in sorted(all_uids):
        # Collect predictions per date
        all_dates_data = []
        for entry in date_entries:
            date = entry["date"]
            if uid not in per_date.get(date, {}):
                continue
            d = per_date[date][uid]
            all_dates_data.append({
                "date": date,
                "y_pred": d["y_pred"],
                "probs_cal": d["probs_cal"],
                "tile_quality_ok": d["tile_quality_ok"],
                "quality_ok_crop": d["quality_ok_crop"],
                "quality_ok_full": d["tile_quality_ok"] and d["quality_ok_crop"],
            })

        # Method 1 & 2: use tile-quality-ok dates (tile-level only, original behaviour)
        usable_12 = [d for d in all_dates_data if d["tile_quality_ok"]]
        # Method 3: additionally require crop-level quality (tile + crop)
        usable_3 = [d for d in all_dates_data if d["quality_ok_full"]]

        # ── M1 fix: do NOT fall back to rejected dates ────────────────────────
        # Previously a fallback used all rejected dates when every tile failed.
        # This silently produced predictions from unusable imagery.  The correct
        # behaviour is to return NOT_IDENTIFIABLE (-1) in that case.
        if not usable_12:
            m1_probs = [0.25] * N_CLASSES
            m1_class = NOT_IDENTIFIABLE
        else:
            m1_probs = avg_probs([d["probs_cal"] for d in usable_12 if d["probs_cal"]])
            m1_class = safe_argmax(m1_probs)

        # Method 2: majority vote (tile-quality-filtered, tie-break by prob_avg)
        if usable_12:
            m2_labels = [d["y_pred"] for d in usable_12]
            m2_class = majority_vote(m2_labels, tiebreak_probs=m1_probs)
        else:
            m2_class = NOT_IDENTIFIABLE

        # Method 3: quality-filtered probability averaging (tile + crop)
        usable_3_valid = [d for d in usable_3 if d["probs_cal"]]
        if usable_3_valid:
            m3_probs = avg_probs([d["probs_cal"] for d in usable_3_valid])
            m3_class = safe_argmax(m3_probs)
        else:
            # Fallback to M1 (not to rejected dates)
            m3_probs = m1_probs
            m3_class = m1_class

        # ── Method 1b: per-building coverage-aware aggregation ─────────────────
        # For each date, a building is "usable" only when:
        #   (a) the tile passes the global quality filter (tile_quality_ok=True), AND
        #   (b) the building's own crop has sufficient valid pixels (quality_ok_crop=True,
        #       i.e. ≤50% zero/nodata pixels in the 256×256 crop window).
        #
        # Rationale: a building whose footprint falls in a nodata strip of a particular
        # post-disaster image should not contribute a damage prediction for that date —
        # the model would be inferring from noise/zeros, not real post-event signal.
        #
        # If NO date satisfies both conditions the building is marked NOT_IDENTIFIABLE
        # (-1 / "not identifiable") rather than being forced into a damage class.
        usable_1b = [d for d in all_dates_data
                     if d["tile_quality_ok"] and d["quality_ok_crop"]]

        if not usable_1b:
            # No post-disaster image with valid coverage for this building footprint.
            m1b_probs = [0.25] * N_CLASSES
            m1b_class = NOT_IDENTIFIABLE
        else:
            m1b_probs = avg_probs([d["probs_cal"] for d in usable_1b if d["probs_cal"]])
            m1b_class = safe_argmax(m1b_probs)

        # ── Method 2b: coverage-aware majority vote (real-world rule) ─────────
        # Identifiability-first, then damage aggregation:
        #   Step 1: Determine valid dates — building must be sufficiently captured
        #           (tile_quality_ok=True AND crop_quality_ok=True).
        #   Step 2: If zero valid dates → NOT_IDENTIFIABLE (building never captured).
        #   Step 3: Majority vote across per-date damage labels from valid dates only.
        #   Tie-break: highest damage class wins (conservative — assume worst plausible
        #   outcome when evidence is split, since under-estimating damage is riskier
        #   than over-estimating in disaster response).
        #
        # This is the authoritative rule for real-world multi-image post-disaster
        # assessment (e.g. LA fire workflow).
        if not usable_1b:
            m2b_class = NOT_IDENTIFIABLE
            m2b_n_valid = 0
            m2b_vote_counts = {}
        else:
            m2b_labels = [d["y_pred"] for d in usable_1b]
            m2b_vote_counts = dict(Counter(m2b_labels))
            m2b_n_valid = len(m2b_labels)
            # Majority vote with max-class tie-break
            m2b_class = majority_vote(m2b_labels, tiebreak_probs=None)
            # tiebreak_probs=None → falls through to min(candidates) in majority_vote,
            # but we want max(candidates) for conservative tie-break. Override here:
            counts = Counter(m2b_labels)
            max_count = max(counts.values())
            candidates = [lbl for lbl, cnt in counts.items() if cnt == max_count]
            if len(candidates) > 1:
                m2b_class = max(candidates)  # conservative: highest damage class wins

        # ── Agreement metrics ─────────────────────────────────────────────────
        all_labels = [d["y_pred"] for d in all_dates_data if d["y_pred"] >= 0]
        used_labels_12 = [d["y_pred"] for d in usable_12 if d["y_pred"] >= 0]
        lbl_entropy = label_entropy(used_labels_12)
        is_unstable = len(set(used_labels_12)) > 1 if used_labels_12 else False

        per_date_summary = {
            d["date"]: {
                "y_pred": d["y_pred"],
                "probs_cal": [round(p, 4) for p in d["probs_cal"]] if d["probs_cal"] else [],
                "tile_quality_ok": d["tile_quality_ok"],
                "quality_ok_crop": d["quality_ok_crop"],
            }
            for d in all_dates_data
        }

        rec = {
            "bldg_uid": uid,
            "n_dates_total": len(all_dates_data),
            "n_dates_used_m1m2": len(usable_12),
            "n_dates_used_m3": len(usable_3_valid),
            "n_dates_valid_coverage": len(usable_1b),   # dates with tile+crop quality OK
            "dates_all": [d["date"] for d in all_dates_data],
            "dates_used_m1m2": [d["date"] for d in usable_12],
            "dates_used_m3": [d["date"] for d in usable_3_valid],
            "dates_used_m1b": [d["date"] for d in usable_1b],
            "dates_tile_rejected": [d["date"] for d in all_dates_data if not d["tile_quality_ok"]],
            "dates_coverage_invalid": [d["date"] for d in all_dates_data
                                       if not d["quality_ok_crop"]],
            # Method 1: prob avg (tile-quality-filtered; -1 if all tiles rejected)
            "m1_prob_avg_class": m1_class,
            "m1_prob_avg_probs": [round(p, 4) for p in m1_probs],
            # Method 2: majority vote (tile-quality-filtered)
            "m2_majority_class": m2_class,
            # Method 3: quality-filtered prob avg (tile + crop)
            "m3_quality_filtered_class": m3_class,
            "m3_quality_filtered_probs": [round(p, 4) for p in m3_probs],
            # Method 1b: per-building coverage-aware (-1 = "not identifiable")
            "m1b_coverage_class": m1b_class,
            "m1b_coverage_probs": [round(p, 4) for p in m1b_probs],
            # Method 2b: coverage-aware majority vote (real-world rule)
            "m2b_coverage_vote_class": m2b_class,
            "m2b_n_valid_dates": m2b_n_valid,
            "m2b_vote_counts": m2b_vote_counts,
            # Agreement
            "label_entropy": lbl_entropy,
            "is_unstable": is_unstable,
            "all_date_labels": all_labels,
            "per_date": per_date_summary,
        }
        results.append(rec)

    # ── Write JSONL ───────────────────────────────────────────────────────────
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")

    # ── Write flat CSV ────────────────────────────────────────────────────────
    csv_cols = [
        "bldg_uid", "n_dates_total", "n_dates_used_m1m2", "n_dates_used_m3",
        "n_dates_valid_coverage",
        "dates_used_m1m2", "dates_tile_rejected", "dates_coverage_invalid",
        "m1_prob_avg_class", "m1_prob_avg_probs",
        "m2_majority_class",
        "m3_quality_filtered_class",
        "m1b_coverage_class", "m1b_coverage_probs",
        "m2b_coverage_vote_class", "m2b_n_valid_dates", "m2b_vote_counts",
        "label_entropy", "is_unstable", "all_date_labels",
    ]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        writer.writeheader()
        for rec in results:
            flat = dict(rec)
            flat["dates_used_m1m2"] = "|".join(flat["dates_used_m1m2"])
            flat["dates_tile_rejected"] = "|".join(flat["dates_tile_rejected"])
            flat["dates_coverage_invalid"] = "|".join(flat.get("dates_coverage_invalid", []))
            flat["m1b_coverage_probs"] = str(flat["m1b_coverage_probs"])
            flat["m1_prob_avg_probs"] = str(flat["m1_prob_avg_probs"])
            flat["m2b_vote_counts"] = str(flat.get("m2b_vote_counts", {}))
            flat["all_date_labels"] = str(flat["all_date_labels"])
            writer.writerow(flat)

    print(f"[done] instances={len(results)}")
    unstable = sum(1 for r in results if r["is_unstable"])
    not_id_m1b = sum(1 for r in results if r["m1b_coverage_class"] == NOT_IDENTIFIABLE)
    not_id_m2b = sum(1 for r in results if r["m2b_coverage_vote_class"] == NOT_IDENTIFIABLE)
    print(f"[summary] unstable={unstable}/{len(results)}")
    print(f"[summary] not_identifiable_m1b={not_id_m1b}/{len(results)}")
    print(f"[summary] not_identifiable_m2b={not_id_m2b}/{len(results)}")
    # M2b damage distribution
    m2b_dist = Counter(r["m2b_coverage_vote_class"] for r in results)
    print(f"[summary] m2b_distribution={dict(sorted(m2b_dist.items()))}")
    # Date rejection summary
    all_dates = list(date_tile_ok.keys())
    for date in all_dates:
        ok = date_tile_ok[date]
        print(f"[summary] date={date} tile_ok={ok}")

    print(f"[done] wrote {args.out_jsonl}")
    print(f"[done] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
