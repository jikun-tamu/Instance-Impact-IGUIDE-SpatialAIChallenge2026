#!/usr/bin/env python3
"""Attach precomputed context ring masks to a Stage-2a manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from PIL import Image


def _disk_kernel(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return ((xx * xx + yy * yy) <= (radius * radius)).astype(np.uint8)


def _resolve_dilate_backend(choice: str) -> tuple[str, Callable[[np.ndarray, int], np.ndarray]]:
    choice = choice.lower()
    if choice in ("auto", "scipy_edt"):
        try:
            from scipy import ndimage as ndi  # type: ignore

            def dilate_scipy_edt(mask01: np.ndarray, radius: int) -> np.ndarray:
                if radius <= 0:
                    return mask01.astype(np.uint8)
                dist_to_mask = ndi.distance_transform_edt(mask01 == 0)
                return (dist_to_mask <= float(radius)).astype(np.uint8)

            return "scipy_edt", dilate_scipy_edt
        except Exception:
            if choice == "scipy_edt":
                raise RuntimeError("Requested scipy_edt backend, but scipy is unavailable.")

    if choice in ("auto", "opencv"):
        try:
            import cv2  # type: ignore

            def dilate_opencv(mask01: np.ndarray, radius: int) -> np.ndarray:
                if radius <= 0:
                    return mask01.astype(np.uint8)
                kernel = _disk_kernel(radius)
                out = cv2.dilate(mask01.astype(np.uint8), kernel, iterations=1)
                return (out > 0).astype(np.uint8)

            return "opencv", dilate_opencv
        except Exception:
            if choice == "opencv":
                raise RuntimeError("Requested opencv backend, but cv2 is unavailable.")

    if choice in ("auto", "scipy"):
        try:
            from scipy import ndimage as ndi  # type: ignore

            def dilate_scipy(mask01: np.ndarray, radius: int) -> np.ndarray:
                if radius <= 0:
                    return mask01.astype(np.uint8)
                out = ndi.binary_dilation(mask01.astype(bool), structure=_disk_kernel(radius).astype(bool))
                return out.astype(np.uint8)

            return "scipy", dilate_scipy
        except Exception:
            if choice == "scipy":
                raise RuntimeError("Requested scipy backend, but scipy is unavailable.")

    def dilate_numpy_square(mask01: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0:
            return mask01.astype(np.uint8)
        padded = np.pad(mask01.astype(np.uint8), radius, mode="constant")
        out = np.zeros_like(mask01, dtype=np.uint8)
        for dy in range(2 * radius + 1):
            for dx in range(2 * radius + 1):
                out |= padded[dy : dy + mask01.shape[0], dx : dx + mask01.shape[1]]
        return out

    return "numpy_square", dilate_numpy_square


def _read_rows(path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def _write_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute Stage-2a ring masks and write a manifest with mask_R.")
    p.add_argument("--labels_csv", type=Path, required=True)
    p.add_argument("--out_manifest", type=Path, required=True)
    p.add_argument("--ring_dir", type=Path, required=True)
    p.add_argument("--mask_col", type=str, default="mask_path")
    p.add_argument("--ring_col", type=str, default="mask_R")
    p.add_argument("--ring_radius_px", type=int, default=48)
    p.add_argument("--backend", choices=["auto", "scipy_edt", "opencv", "scipy", "numpy_square"], default="auto")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--progress_every", type=int, default=1000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows, fieldnames = _read_rows(args.labels_csv)
    if args.ring_col not in fieldnames:
        fieldnames.append(args.ring_col)
    args.ring_dir.mkdir(parents=True, exist_ok=True)
    backend, dilate = _resolve_dilate_backend(args.backend)
    written = 0
    reused = 0
    empty = 0
    for idx, row in enumerate(rows):
        mask_path = Path(row.get(args.mask_col, "") or "")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask path at row {idx}: {mask_path}")
        ring_path = args.ring_dir / f"{idx:08d}_{mask_path.stem}_r{args.ring_radius_px}.png"
        if ring_path.exists() and not args.overwrite:
            reused += 1
        else:
            mask = (np.asarray(Image.open(mask_path).convert("L")) > 0).astype(np.uint8)
            if mask.sum() == 0:
                ring = np.zeros_like(mask, dtype=np.uint8)
                empty += 1
            else:
                dilated = dilate(mask, int(args.ring_radius_px))
                ring = np.clip(dilated - mask, 0, 1).astype(np.uint8)
            Image.fromarray((ring * 255).astype(np.uint8), mode="L").save(ring_path)
            written += 1
        row[args.ring_col] = str(ring_path.resolve())
        if args.progress_every > 0 and (idx + 1) % args.progress_every == 0:
            print(
                "[stage2a_ring_manifest]",
                f"{idx + 1}/{len(rows)}",
                f"backend={backend}",
                f"written={written}",
                f"reused={reused}",
                f"empty={empty}",
                flush=True,
            )
    _write_rows(args.out_manifest, rows, fieldnames)
    print("[done] backend=", backend)
    print("[done] rows=", len(rows))
    print("[done] written=", written, "reused=", reused, "empty=", empty)
    print("[done] wrote=", args.out_manifest)


if __name__ == "__main__":
    main()
