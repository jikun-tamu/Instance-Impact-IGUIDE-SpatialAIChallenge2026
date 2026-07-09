#!/usr/bin/env python3
"""Visualize instance-level predictions on post-disaster crops.

Reads inference JSONL outputs (from infer_stage2_ensemble.py), joins to stage2 CSV
for image/mask paths, overlays mask_M as a semi-transparent filled region colored by
predicted label, and draws multi-line metric panels:
- top-left: Stage2b metrics
- bottom-left: Stage1 confidence
- bottom-right: Stage2a metrics (when --stage2a_csv is provided)
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PALETTE = {
    0: (0, 180, 0),     # green
    1: (170, 200, 0),   # yellow-green
    2: (245, 155, 0),   # orange
    3: (220, 40, 40),   # red
}


def parse_args():
    p = argparse.ArgumentParser(description="Overlay Stage-2 prediction metrics on post crops.")
    p.add_argument("--pred_jsonl", type=Path, required=True, help="Path to inference JSONL.")
    p.add_argument("--csv", type=Path, required=True, help="Path to stage2_samples.csv (for image/mask paths).")
    p.add_argument(
        "--stage2a_csv",
        type=Path,
        default=Path(""),
        help="Optional Stage2a prediction CSV (from infer_stage2a.py) for pop/type/conf overlays.",
    )
    p.add_argument("--out_dir", type=Path, default=Path("outputs/vis_predictions_overlay"))
    p.add_argument("--max_outputs", type=int, default=100, help="Max number of visualizations to write.")
    p.add_argument(
        "--gt_labels",
        type=str,
        default="",
        help="Optional comma-separated GT labels to keep (e.g. 2 or 2,3). Empty => all.",
    )
    p.add_argument("--line_width", type=int, default=3, help="Contour line width in pixels.")
    p.add_argument("--fill_opacity", type=float, default=0.5, help="Mask fill opacity in [0,1].")
    return p.parse_args()


def parse_label_filter(raw):
    if not raw.strip():
        return None
    vals = set()
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        vals.add(int(x))
    return vals


def key_from_row(row):
    return (row.get("event_id", ""), row.get("tile_id", ""), row.get("bldg_uid", ""))


def load_csv_index(csv_path):
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    idx_key = {}
    for r in rows:
        idx_key[key_from_row(r)] = r
    return rows, idx_key


def read_jsonl(path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(json.loads(s))
    return out


def mask_edge(mask_bin):
    # 8-neighborhood edge: pixel is edge if any neighbor is zero.
    m = mask_bin.astype(np.uint8)
    if m.ndim != 2:
        raise ValueError("Expected 2D mask")
    p = np.pad(m, 1, mode="constant", constant_values=0)
    c = p[1:-1, 1:-1]
    n0 = p[:-2, :-2]
    n1 = p[:-2, 1:-1]
    n2 = p[:-2, 2:]
    n3 = p[1:-1, :-2]
    n4 = p[1:-1, 2:]
    n5 = p[2:, :-2]
    n6 = p[2:, 1:-1]
    n7 = p[2:, 2:]
    interior = c & n0 & n1 & n2 & n3 & n4 & n5 & n6 & n7
    edge = (c == 1) & (interior == 0)
    return edge


def dilate_binary(mask, iters=1):
    m = mask.astype(np.uint8)
    for _ in range(max(0, int(iters))):
        p = np.pad(m, 1, mode="constant", constant_values=0)
        cands = [
            p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:],
            p[1:-1, :-2], p[1:-1, 1:-1], p[1:-1, 2:],
            p[2:, :-2], p[2:, 1:-1], p[2:, 2:],
        ]
        m = np.maximum.reduce(cands)
    return m.astype(bool)


def overlay_mask_fill(base_rgb, mask_path, color, fill_opacity=0.5):
    arr = np.array(base_rgb.convert("RGB"), dtype=np.uint8)
    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
    out = arr.astype(np.float32).copy()
    alpha = float(np.clip(fill_opacity, 0.0, 1.0))
    c = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    # Filled polygon overlay only (no boundary stroke).
    out[mask] = (1.0 - alpha) * out[mask] + alpha * c
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def draw_panel(img, lines, fg=(0, 255, 0), bg=(0, 0, 0, 170), pad=4, anchor="tl"):
    draw = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.load_default()
    if not lines:
        return img
    widths = []
    heights = []
    for ln in lines:
        bb = draw.textbbox((0, 0), ln, font=font)
        widths.append(bb[2] - bb[0])
        heights.append(bb[3] - bb[1])
    tw = max(widths)
    line_h = max(heights)
    th = len(lines) * line_h + (len(lines) - 1) * 2
    W, H = img.size
    # Anchor panel in image corners.
    if anchor == "tl":
        x0, y0 = 0, 0
    elif anchor == "tr":
        x0, y0 = max(0, W - (tw + 2 * pad)), 0
    elif anchor == "bl":
        x0, y0 = 0, max(0, H - (th + 2 * pad))
    elif anchor == "br":
        x0, y0 = max(0, W - (tw + 2 * pad)), max(0, H - (th + 2 * pad))
    else:
        x0, y0 = 0, 0
    x1, y1 = x0 + tw + 2 * pad, y0 + th + 2 * pad
    draw.rectangle([x0, y0, x1, y1], fill=bg)
    y = y0 + pad
    for ln in lines:
        draw.text((x0 + pad, y), ln, fill=fg, font=font)
        y += line_h + 2
    return img


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gt_filter = parse_label_filter(args.gt_labels)

    csv_rows, csv_idx = load_csv_index(args.csv)
    preds = read_jsonl(args.pred_jsonl)
    s2a_idx = {}
    if str(args.stage2a_csv):
        with args.stage2a_csv.open("r", encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f)
            for r in rd:
                uid = r.get("building_uid", "")
                if uid:
                    s2a_idx[uid] = r
    print("Loaded predictions:", len(preds))
    print("Loaded CSV index rows:", len(csv_rows))
    if s2a_idx:
        print("Loaded Stage2a rows:", len(s2a_idx))
    if gt_filter is not None:
        print("GT filter:", sorted(gt_filter))

    written = 0
    skipped_missing = 0
    skipped_filter = 0

    for i, rec in enumerate(preds):
        y_true = int(rec.get("y_true", -1))
        if gt_filter is not None and y_true not in gt_filter:
            skipped_filter += 1
            continue

        row = None
        # Prefer exact row-position join when sample_index is present in JSONL.
        sidx = rec.get("sample_index", None)
        if isinstance(sidx, int) and 0 <= sidx < len(csv_rows):
            row = csv_rows[sidx]
        if row is None:
            # Fallback to identity-key join.
            k = (rec.get("event_id", ""), rec.get("tile_id", ""), rec.get("bldg_uid", ""))
            row = csv_idx.get(k)
        if row is None:
            skipped_missing += 1
            continue

        post_path = row.get("post_crop", "")
        mask_path = row.get("mask_M", "")
        if not post_path or not mask_path or not Path(post_path).exists() or not Path(mask_path).exists():
            skipped_missing += 1
            continue

        pred_cls = int(rec.get("y_pred_ensemble", -1))
        color = PALETTE.get(pred_cls, (255, 255, 255))
        base = Image.open(post_path).convert("RGB")
        vis = overlay_mask_fill(
            base,
            mask_path,
            color=color,
            fill_opacity=args.fill_opacity,
        )

        pmax = float(rec.get("pmax", 0.0))
        margin = float(rec.get("margin", 0.0))
        entropy = float(rec.get("entropy", 0.0))
        exp_sev = float(rec.get("expected_severity_ensemble", 0.0))
        var_exp = float(rec.get("var_expected_severity_weighted", 0.0))

        lines = [
            f"pred: {pred_cls} | gt: {y_true}",
            f"exp severity: {exp_sev:.3f} ({var_exp:.3f}) | pmax: {pmax:.3f}",
            f"margin: {margin:.3f} | entropy: {entropy:.3f}",
        ]
        vis = draw_panel(vis, lines, anchor="tl")

        # Stage1 confidence from shared/stage2 csv row.
        s1_conf = row.get("sam3_confidence", "")
        try:
            s1_conf_s = f"s1_conf: {float(s1_conf):.3f}"
        except Exception:
            s1_conf_s = "s1_conf: n/a"
        vis = draw_panel(vis, [s1_conf_s], anchor="bl")

        # Stage2a summary from optional csv.
        uid = rec.get("bldg_uid", "")
        r2a = s2a_idx.get(uid, {})
        if r2a:
            try:
                pop_s = f"pop: {float(r2a.get('pred_population', 0.0)):.2f}"
            except Exception:
                pop_s = "pop: n/a"
            t_s = f"type: {r2a.get('pred_type_class', 'n/a')}"
            try:
                c_s = f"s2a_conf: {float(r2a.get('pred_type_conf', 0.0)):.3f}"
            except Exception:
                c_s = "s2a_conf: n/a"
            vis = draw_panel(vis, [pop_s, t_s, c_s], anchor="br")

        safe_evt = rec.get("event_id", "evt").replace("/", "_")
        safe_tile = rec.get("tile_id", "tile").replace("/", "_")
        safe_uid = rec.get("bldg_uid", str(i)).replace("/", "_")
        out_name = f"{written:06d}_{safe_evt}_{safe_tile}_{safe_uid}.png"
        vis.save(args.out_dir / out_name)
        written += 1
        if written >= args.max_outputs:
            break

    print("Wrote overlays:", written)
    print("Skipped by gt filter:", skipped_filter)
    print("Skipped missing row/path:", skipped_missing)
    print("Output dir:", args.out_dir)


if __name__ == "__main__":
    main()
