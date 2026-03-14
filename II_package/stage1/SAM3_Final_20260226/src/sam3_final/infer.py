from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import login
from samgeo import SamGeo3

from .utils import ensure_dir


@dataclass
class Sam3Config:
    backend: str = "meta"
    device: str | None = None
    checkpoint_path: str | None = None
    load_from_hf: bool = True
    hf_token: str | None = None


@dataclass
class InferenceResult:
    mask_path: Path | None
    score_path: Path | None
    ann_path: Path | None
    num_instances: int
    t_infer_s: float
    t_save_s: float


def init_sam3(cfg: Sam3Config) -> SamGeo3:
    if cfg.hf_token:
        login(token=cfg.hf_token, add_to_git_credential=False)
    return SamGeo3(
        backend=cfg.backend,
        device=cfg.device,
        checkpoint_path=cfg.checkpoint_path,
        load_from_HF=cfg.load_from_hf,
    )


def clear_gpu_cache() -> None:
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def infer_single_image(
    sam3: SamGeo3,
    image_path: str | Path,
    output_dir: str | Path,
    prompt: str = "building",
    min_size: int = 100,
    save_masks: bool = True,
    save_scores: bool = True,
    save_ann: bool = True,
    dpi: int = 150,
    font_size: int = 8,
) -> InferenceResult | None:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    mask_dir = ensure_dir(output_dir / "masks")
    ann_dir = ensure_dir(output_dir / "annotations")

    import time

    sam3.set_image(str(image_path))
    t0 = time.perf_counter()
    sam3.generate_masks(prompt=prompt, min_size=min_size)
    t1 = time.perf_counter()

    if not hasattr(sam3, "masks") or len(sam3.masks) == 0:
        return None

    mask_path = None
    score_path = None
    t2 = time.perf_counter()
    if save_masks:
        mask_path = mask_dir / f"{image_path.stem}.tif"
        score_path = mask_dir / f"{image_path.stem}_scores.tif" if save_scores else None
        if save_scores:
            sam3.save_masks(output=str(mask_path), save_scores=str(score_path), unique=True)
        else:
            sam3.save_masks(output=str(mask_path), unique=True)

    ann_path = None
    if save_ann:
        ann_path = ann_dir / f"{image_path.stem}_ann.png"
        sam3.show_anns(output=str(ann_path), dpi=dpi, font_size=font_size)
    t3 = time.perf_counter()

    return InferenceResult(
        mask_path=mask_path,
        score_path=score_path,
        ann_path=ann_path,
        num_instances=len(sam3.masks),
        t_infer_s=t1 - t0,
        t_save_s=t3 - t2,
    )
