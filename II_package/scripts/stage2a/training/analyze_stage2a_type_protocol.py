#!/usr/bin/env python3
"""Audit Stage-2a building-type protocol and prediction slices."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from PIL import Image, ImageDraw

try:
    from scripts.stage2a.common import (
        CLASS_NAMES,
        classification_metrics,
        effective_split_group,
        metrics_from_prediction_rows,
        read_csv_rows,
        resolve_path,
        safe_float,
        safe_int,
        save_json,
        split_has_leakage,
        tile_train_val_test_split,
    )
except ImportError:  # pragma: no cover
    from II_package.scripts.stage2a.common import (
        CLASS_NAMES,
        classification_metrics,
        effective_split_group,
        metrics_from_prediction_rows,
        read_csv_rows,
        resolve_path,
        safe_float,
        safe_int,
        save_json,
        split_has_leakage,
        tile_train_val_test_split,
    )


def _label_name(row: Mapping[str, object]) -> str:
    name = str(row.get("type_class_name", "") or row.get("true_type_class", "")).strip()
    if name:
        return name
    idx = safe_int(row.get("type_class", row.get("true_type_idx")), -1)
    if 0 <= idx < len(CLASS_NAMES):
        return CLASS_NAMES[idx]
    return "__missing__"


def _counter(rows: Sequence[Mapping[str, object]], col: str) -> Dict[str, int]:
    return dict(Counter(str(r.get(col, "") or "__missing__") for r in rows))


def _class_counts(rows: Sequence[Mapping[str, object]]) -> Dict[str, int]:
    counts = Counter(_label_name(r) for r in rows)
    return {name: int(counts.get(name, 0)) for name in [*CLASS_NAMES, "__missing__"]}


def _nested_counts(rows: Sequence[Mapping[str, object]], outer_col: str) -> Dict[str, Dict[str, int]]:
    groups: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(outer_col, "") or "__missing__")].append(row)
    return {group: _class_counts(items) for group, items in sorted(groups.items())}


def _effective_group_counts(rows: Sequence[Mapping[str, object]], split_indices: Sequence[int], split_group_col: str) -> Dict[str, int]:
    return dict(Counter(effective_split_group(rows[i], split_group_col, i) for i in split_indices))


def _prediction_group_metrics(rows: Sequence[Mapping[str, object]], group_col: str) -> Dict[str, object]:
    groups: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_col, "") or "__missing__")].append(row)
    out: Dict[str, object] = {}
    for group, items in sorted(groups.items()):
        if not items:
            continue
        out[group] = {
            "support": len(items),
            "metrics": metrics_from_prediction_rows(items),
        }
    return out


def audit_protocol(
    rows: Sequence[Mapping[str, object]],
    split_group_col: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, object]:
    split = tile_train_val_test_split(
        rows,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        tile_col=split_group_col,
    )
    split_rows = {name: [rows[i] for i in idxs] for name, idxs in split.items()}
    split_groups = {
        name: sorted({effective_split_group(rows[i], split_group_col, i) for i in idxs})
        for name, idxs in split.items()
    }
    return {
        "seed": seed,
        "split_group_col": split_group_col,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "row_count": len(rows),
        "split_counts": {name: len(items) for name, items in split_rows.items()},
        "group_counts": {name: len(groups) for name, groups in split_groups.items()},
        "split_leakage": split_has_leakage(rows, split, split_group_col),
        "effective_group_fallback": f"{split_group_col} -> tile_id -> __row_i",
        "raw_missing_group_counts": {
            name: sum(1 for i in idxs if not str(rows[i].get(split_group_col, "")))
            for name, idxs in split.items()
        },
        "effective_group_counts_by_split": {
            name: _effective_group_counts(rows, idxs, split_group_col)
            for name, idxs in split.items()
        },
        "class_counts_by_split": {name: _class_counts(items) for name, items in split_rows.items()},
        "classification_source_counts_by_split": {name: _counter(items, "classification_source") for name, items in split_rows.items()},
        "class_by_classification_source": {name: _nested_counts(items, "classification_source") for name, items in split_rows.items()},
        "geoid_counts_by_split": {name: _counter(items, "GEOID") for name, items in split_rows.items()},
        "population_mean_by_class": {
            class_name: float(
                sum(safe_float(r.get("estimated_population"), 0.0) for r in rows if _label_name(r) == class_name)
                / max(1, sum(1 for r in rows if _label_name(r) == class_name))
            )
            for class_name in CLASS_NAMES
        },
    }


def _panel_thumb(row: Mapping[str, object], base_dir: Path, size: int) -> Image.Image | None:
    crop_path = str(row.get("crop_path", "") or "")
    if not crop_path:
        return None
    path = resolve_path(crop_path, base_dir)
    if not path.exists():
        return None
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    img.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size + 18), (255, 255, 255))
    x = (size - img.width) // 2
    canvas.paste(img, (x, 0))
    draw = ImageDraw.Draw(canvas)
    uid = str(row.get("building_uid", ""))[-10:]
    draw.text((3, size + 3), uid, fill=(0, 0, 0))
    return canvas


def write_visual_panels(
    rows: Sequence[Mapping[str, object]],
    split: Mapping[str, Sequence[int]],
    split_group_col: str,
    out_dir: Path,
    base_dir: Path,
    panel_classes: Sequence[str],
    max_per_class: int,
    thumb_size: int = 112,
) -> Dict[str, object]:
    panel_dir = out_dir / "visual_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, object] = {}
    for split_name, idxs in split.items():
        for class_name in panel_classes:
            selected = [rows[i] for i in idxs if _label_name(rows[i]) == class_name][: max(0, max_per_class)]
            thumbs = [_panel_thumb(row, base_dir, thumb_size) for row in selected]
            thumbs = [thumb for thumb in thumbs if thumb is not None]
            if not thumbs:
                manifest[f"{split_name}:{class_name}"] = {"count": len(selected), "panel": None}
                continue
            cols = min(4, len(thumbs))
            rows_n = (len(thumbs) + cols - 1) // cols
            panel = Image.new("RGB", (cols * thumb_size, rows_n * (thumb_size + 18)), (240, 240, 240))
            for j, thumb in enumerate(thumbs):
                panel.paste(thumb, ((j % cols) * thumb_size, (j // cols) * (thumb_size + 18)))
            out_path = panel_dir / f"{split_name}_{class_name}.jpg"
            panel.save(out_path, quality=90)
            manifest[f"{split_name}:{class_name}"] = {
                "count": len(selected),
                "rendered": len(thumbs),
                "panel": str(out_path),
                "split_group_col": split_group_col,
            }
    return manifest


def prediction_report(prediction_csv: Path) -> Dict[str, object]:
    rows = read_csv_rows(prediction_csv)
    y_true = [safe_int(r.get("true_type_idx"), -1) for r in rows]
    y_pred = [safe_int(r.get("pred_type_idx"), -1) for r in rows]
    valid = [(t, p, r) for t, p, r in zip(y_true, y_pred, rows) if t >= 0 and p >= 0]
    valid_rows = [r for _, _, r in valid]
    return {
        "prediction_csv": str(prediction_csv),
        "row_count": len(rows),
        "valid_type_rows": len(valid_rows),
        "overall": metrics_from_prediction_rows(valid_rows) if valid_rows else {},
        "classification_by_source": _prediction_group_metrics(valid_rows, "classification_source"),
        "classification_by_true_type": _prediction_group_metrics(valid_rows, "true_type_class"),
        "argmax_metrics": classification_metrics([t for t, _, _ in valid], y_pred=[p for _, p, _ in valid])
        if valid
        else {},
    }


def write_markdown(report: Mapping[str, object], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage2a Type Protocol Audit",
        "",
        f"- rows: `{report.get('row_count')}`",
        f"- split group: `{report.get('split_group_col')}`",
        f"- split leakage: `{report.get('split_leakage')}`",
        f"- effective group fallback: `{report.get('effective_group_fallback')}`",
        f"- raw missing group counts: `{report.get('raw_missing_group_counts')}`",
        f"- split counts: `{report.get('split_counts')}`",
        f"- group counts: `{report.get('group_counts')}`",
        "",
        "## Class Counts By Split",
        "",
    ]
    for split, counts in dict(report.get("class_counts_by_split", {})).items():
        lines.append(f"### {split}")
        for name, count in dict(counts).items():
            lines.append(f"- `{name}`: {count}")
        lines.append("")
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit Stage-2a type-class protocol and prediction slices.")
    p.add_argument("--labels_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--split_group_col", type=str, default="GEOID")
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--prediction_csv", type=Path, action="append", default=[])
    p.add_argument("--base_dir", type=Path, default=Path("."))
    p.add_argument("--panel_classes", type=str, default="institutional,other")
    p.add_argument("--panel_max_per_class", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv_rows(args.labels_csv)
    report = audit_protocol(rows, args.split_group_col, args.val_ratio, args.test_ratio, args.seed)
    split = tile_train_val_test_split(
        rows,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        tile_col=args.split_group_col,
    )
    panel_classes = [x.strip() for x in args.panel_classes.split(",") if x.strip()]
    report["visual_panels"] = write_visual_panels(
        rows,
        split,
        args.split_group_col,
        args.out_dir,
        args.base_dir,
        panel_classes,
        args.panel_max_per_class,
    )
    report["prediction_reports"] = [prediction_report(p) for p in args.prediction_csv]
    save_json(args.out_dir / "type_protocol_audit.json", report)
    write_markdown(report, args.out_dir / "type_protocol_audit.md")
    print("[done] wrote", args.out_dir / "type_protocol_audit.json")


if __name__ == "__main__":
    main()
