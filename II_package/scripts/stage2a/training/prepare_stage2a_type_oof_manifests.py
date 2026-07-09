#!/usr/bin/env python3
"""Create grouped out-of-fold manifests for Stage-2a type feature generation."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build tile/group-blocked OOF manifests from a fixed train/val Stage2a manifest.")
    p.add_argument("--labels_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--out_prefix", type=str, default="labels_manifest_m21_oof")
    p.add_argument("--source_split_col", type=str, default="m20_split")
    p.add_argument("--source_train_value", type=str, default="train")
    p.add_argument("--fold_split_col", type=str, default="m21_oof_split")
    p.add_argument("--fold_id_col", type=str, default="m21_oof_fold")
    p.add_argument("--group_col", type=str, default="tile_base")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
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


def effective_group(row: Mapping[str, object], group_col: str, row_idx: int) -> str:
    return str(row.get(group_col) or row.get("tile_id") or f"__row_{row_idx}")


def split_counts(rows: Sequence[Mapping[str, object]], split_col: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(row.get(split_col, "") or "__missing__") for row in rows).items()))


def assign_group_folds(
    rows: Sequence[Mapping[str, str]],
    train_indices: Sequence[int],
    group_col: str,
    n_folds: int,
    seed: int,
) -> Dict[str, int]:
    group_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx in train_indices:
        group_to_indices[effective_group(rows[idx], group_col, idx)].append(idx)
    groups = list(group_to_indices)
    rng = random.Random(seed)
    rng.shuffle(groups)
    return {group: i % n_folds for i, group in enumerate(groups)}


def fold_leakage(rows: Sequence[Mapping[str, object]], split_col: str, group_col: str) -> bool:
    groups_by_split: Dict[str, set[str]] = defaultdict(set)
    for i, row in enumerate(rows):
        groups_by_split[str(row.get(split_col, "") or "")].add(effective_group(row, group_col, i))
    train_groups = groups_by_split.get("train", set())
    val_groups = groups_by_split.get("val", set())
    return bool(train_groups & val_groups)


def main() -> None:
    args = parse_args()
    if args.n_folds < 2:
        raise ValueError("--n_folds must be at least 2")
    rows, original_fields = read_csv_rows(args.labels_csv)
    train_indices = [
        i
        for i, row in enumerate(rows)
        if str(row.get(args.source_split_col, "") or "").strip().lower() == args.source_train_value
    ]
    if not train_indices:
        raise RuntimeError(f"No source train rows found in {args.source_split_col}={args.source_train_value!r}")

    group_to_fold = assign_group_folds(rows, train_indices, args.group_col, args.n_folds, args.seed)
    fold_counts = Counter(group_to_fold.values())
    fieldnames = list(original_fields)
    for field in (args.fold_split_col, args.fold_id_col):
        if field not in fieldnames:
            fieldnames.append(field)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    for fold in range(args.n_folds):
        fold_rows: List[MutableMapping[str, object]] = []
        for idx in train_indices:
            src = rows[idx]
            group = effective_group(src, args.group_col, idx)
            assigned_fold = int(group_to_fold[group])
            out = dict(src)
            out[args.fold_id_col] = str(assigned_fold)
            out[args.fold_split_col] = "val" if assigned_fold == fold else "train"
            fold_rows.append(out)
        out_csv = args.out_dir / f"{args.out_prefix}_fold{fold}.csv"
        write_csv_rows(out_csv, fold_rows, fieldnames)
        manifests.append(
            {
                "fold": fold,
                "manifest": str(out_csv),
                "counts": split_counts(fold_rows, args.fold_split_col),
                "leakage": fold_leakage(fold_rows, args.fold_split_col, args.group_col),
            }
        )

    audit = {
        "labels_csv": str(args.labels_csv),
        "out_dir": str(args.out_dir),
        "out_prefix": args.out_prefix,
        "source_split_col": args.source_split_col,
        "source_train_value": args.source_train_value,
        "fold_split_col": args.fold_split_col,
        "fold_id_col": args.fold_id_col,
        "group_col": args.group_col,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "source_split_counts": split_counts(rows, args.source_split_col),
        "train_rows": len(train_indices),
        "train_groups": len(group_to_fold),
        "groups_per_fold": {str(k): int(v) for k, v in sorted(fold_counts.items())},
        "manifests": manifests,
    }
    save_json(args.out_dir / f"{args.out_prefix}_audit.json", audit)
    print("[done] train_rows=", len(train_indices), "train_groups=", len(group_to_fold))
    print("[done] wrote=", args.out_dir / f"{args.out_prefix}_audit.json")


if __name__ == "__main__":
    main()
