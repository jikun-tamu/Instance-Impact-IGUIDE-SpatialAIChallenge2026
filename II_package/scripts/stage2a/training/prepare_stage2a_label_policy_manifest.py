#!/usr/bin/env python3
"""Prepare Stage-2a type-label policy manifests.

The Stage2a type bottleneck is concentrated in two ambiguous minority classes:
``institutional`` and ``other``. This utility creates policy-specific manifests
while preserving a fixed blocked split from the original manifest, so label
merge/drop experiments are comparable instead of being reshuffled after rows
are removed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence


POLICY_DESCRIPTIONS = {
    "merge_institutional_other": "Keep all rows; map institutional labels into the existing other class.",
    "drop_institutional": "Remove institutional rows; train/evaluate on the remaining four-class label space.",
    "drop_other": "Remove other rows; train/evaluate on the remaining four-class label space.",
    "drop_institutional_other": "Remove both institutional and other rows; train/evaluate on the remaining three-class label space.",
}

CLASS_NAMES = ["residential_small", "residential_multi", "commercial", "institutional", "other"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

ACTIVE_CLASSES = {
    "merge_institutional_other": ["residential_small", "residential_multi", "commercial", "other"],
    "drop_institutional": ["residential_small", "residential_multi", "commercial", "other"],
    "drop_other": ["residential_small", "residential_multi", "commercial", "institutional"],
    "drop_institutional_other": ["residential_small", "residential_multi", "commercial"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a Stage2a label-policy manifest with a fixed split column.")
    p.add_argument("--labels_csv", type=Path, required=True)
    p.add_argument("--out_manifest", type=Path, required=True)
    p.add_argument("--out_audit_json", type=Path, default=None)
    p.add_argument("--policy", choices=sorted(POLICY_DESCRIPTIONS), required=True)
    p.add_argument("--split_col", type=str, default="m20_split")
    p.add_argument("--split_group_col", type=str, default="GEOID")
    p.add_argument("--split_seed", type=int, default=2025)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    return p.parse_args()


def read_csv_rows(path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def effective_split_group(row: Mapping[str, object], group_col: str, row_idx: int) -> str:
    return str(row.get(group_col) or row.get("tile_id") or f"__row_{row_idx}")


def tile_train_val_test_split(
    rows: Sequence[Mapping[str, object]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    tile_col: str,
) -> Dict[str, List[int]]:
    if not rows:
        raise ValueError("Cannot split empty rows")
    group_to_idx: Dict[str, List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        group_to_idx[effective_split_group(row, tile_col, i)].append(i)
    groups = list(group_to_idx)
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_groups = len(groups)
    n_test = max(1, int(round(n_groups * test_ratio))) if test_ratio > 0 else 0
    n_val = max(1, int(round(n_groups * val_ratio))) if val_ratio > 0 else 0
    if n_val + n_test >= n_groups:
        if n_test == 0:
            if n_groups < 2:
                raise ValueError("Need at least two groups for train/val split")
            n_val = max(1, min(n_val, n_groups - 1))
        else:
            if n_groups < 3:
                raise ValueError("Need at least three groups for train/val/test split")
            n_test = max(1, min(n_test, n_groups - 2))
            n_val = max(1, min(n_val, n_groups - n_test - 1))
    test_groups = set(groups[:n_test])
    val_groups = set(groups[n_test : n_test + n_val])
    split = {"train": [], "val": [], "test": []}
    for group, idxs in group_to_idx.items():
        if group in test_groups:
            split["test"].extend(idxs)
        elif group in val_groups:
            split["val"].extend(idxs)
        else:
            split["train"].extend(idxs)
    if not split["train"] or not split["val"] or (test_ratio > 0 and not split["test"]):
        raise RuntimeError(f"Invalid split sizes: { {k: len(v) for k, v in split.items()} }")
    return split


def class_name(row: Mapping[str, object]) -> str:
    name = str(row.get("type_class_name", "") or "").strip()
    if name:
        return name
    raw = str(row.get("type_class", "") or "").strip()
    try:
        idx = int(float(raw))
    except ValueError:
        return ""
    return CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else ""


def class_idx(name: str) -> int:
    if name not in CLASS_TO_IDX:
        raise ValueError(f"Unknown class name {name!r}; expected one of {CLASS_NAMES}")
    return int(CLASS_TO_IDX[name])


def original_split_assignments(
    rows: Sequence[Mapping[str, object]],
    split_group_col: str,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
) -> Dict[int, str]:
    split = tile_train_val_test_split(
        rows,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=split_seed,
        tile_col=split_group_col,
    )
    assignments: Dict[int, str] = {}
    for split_name, indices in split.items():
        for idx in indices:
            assignments[int(idx)] = split_name
    return assignments


def transform_row(row: Mapping[str, object], policy: str) -> MutableMapping[str, object] | None:
    original_name = class_name(row)
    if policy == "merge_institutional_other":
        mapped_name = "other" if original_name in {"institutional", "other"} else original_name
    elif policy == "drop_institutional":
        if original_name == "institutional":
            return None
        mapped_name = original_name
    elif policy == "drop_other":
        if original_name == "other":
            return None
        mapped_name = original_name
    elif policy == "drop_institutional_other":
        if original_name in {"institutional", "other"}:
            return None
        mapped_name = original_name
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unsupported policy {policy!r}")

    out: MutableMapping[str, object] = dict(row)
    out["label_policy"] = policy
    out["original_type_class"] = row.get("type_class", "")
    out["original_type_class_name"] = original_name
    out["type_class_name"] = mapped_name
    out["type_class"] = str(class_idx(mapped_name))
    return out


def count_by(rows: Iterable[Mapping[str, object]], key: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "") or "__missing__") for row in rows).items()))


def class_counts(rows: Iterable[Mapping[str, object]], name_col: str = "type_class_name") -> Dict[str, int]:
    counts = Counter(str(row.get(name_col, "") or "__missing__") for row in rows)
    return {name: int(counts.get(name, 0)) for name in CLASS_NAMES if counts.get(name, 0) > 0}


def class_counts_by_split(rows: Sequence[Mapping[str, object]], split_col: str) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for split in ("train", "val", "test"):
        out[split] = class_counts([row for row in rows if row.get(split_col) == split])
    return out


def class_source_counts(rows: Iterable[Mapping[str, object]]) -> Dict[str, Dict[str, int]]:
    nested: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        nested[str(row.get("type_class_name", "") or "__missing__")][str(row.get("classification_source", "") or "__missing__")] += 1
    return {name: dict(sorted(counts.items())) for name, counts in sorted(nested.items())}


def split_leakage(rows: Sequence[Mapping[str, object]], split_col: str, group_col: str) -> bool:
    groups: Dict[str, set[str]] = defaultdict(set)
    for i, row in enumerate(rows):
        split = str(row.get(split_col, "") or "")
        if split:
            groups[split].add(effective_split_group(row, group_col, i))
    keys = list(groups)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if groups[a] & groups[b]:
                return True
    return False


def main() -> None:
    args = parse_args()
    rows, original_fields = read_csv_rows(args.labels_csv)
    assignments = original_split_assignments(rows, args.split_group_col, args.val_ratio, args.test_ratio, args.split_seed)

    out_rows: List[MutableMapping[str, object]] = []
    dropped_original_names: Counter[str] = Counter()
    for i, row in enumerate(rows):
        transformed = transform_row(row, args.policy)
        if transformed is None:
            dropped_original_names[class_name(row)] += 1
            continue
        transformed[args.split_col] = assignments[i]
        out_rows.append(transformed)

    extra_fields = ["label_policy", "original_type_class", "original_type_class_name", args.split_col]
    fieldnames = list(original_fields)
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)
    write_csv_rows(args.out_manifest, out_rows, fieldnames)

    split_counts = count_by(out_rows, args.split_col)
    audit = {
        "policy": args.policy,
        "description": POLICY_DESCRIPTIONS[args.policy],
        "labels_csv": str(args.labels_csv),
        "out_manifest": str(args.out_manifest),
        "split_col": args.split_col,
        "split_group_col": args.split_group_col,
        "split_seed": args.split_seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "active_class_names": ACTIVE_CLASSES[args.policy],
        "rows_before": len(rows),
        "rows_after": len(out_rows),
        "retained_fraction": float(len(out_rows) / max(1, len(rows))),
        "dropped_original_class_counts": dict(sorted(dropped_original_names.items())),
        "original_class_counts": class_counts(rows),
        "policy_class_counts": class_counts(out_rows),
        "policy_original_class_counts": class_counts(out_rows, "original_type_class_name"),
        "split_counts": split_counts,
        "class_counts_by_split": class_counts_by_split(out_rows, args.split_col),
        "classification_source_counts": count_by(out_rows, "classification_source"),
        "class_source_counts": class_source_counts(out_rows),
        "split_leakage": split_leakage(out_rows, args.split_col, args.split_group_col),
    }
    out_audit = args.out_audit_json or args.out_manifest.with_name(f"{args.out_manifest.stem}_policy_audit.json")
    save_json(out_audit, audit)
    print("[done] policy", args.policy)
    print("[done] rows", len(rows), "->", len(out_rows))
    print("[done] wrote", args.out_manifest)
    print("[done] wrote", out_audit)


if __name__ == "__main__":
    main()
