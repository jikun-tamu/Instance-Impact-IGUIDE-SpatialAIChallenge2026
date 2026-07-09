#!/usr/bin/env python3
"""Shared Stage-2a experiment utilities.

This module keeps the legacy EfficientNet-B0 contract intact while adding the
small accuracy-track hooks needed by the revised plan: polygon-derived geometry
features and optional soft type-conditioning for the population head.
"""

from __future__ import annotations

import csv
import json
import math
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms
from torchvision.transforms import functional as TF


CLASS_NAMES = ["residential_small", "residential_multi", "commercial", "institutional", "other"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
PROB_COLUMNS = [f"prob_{name}" for name in CLASS_NAMES]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_GEOMETRY_COLS = [
    "log_footprint_m2",
    "mask_fill_ratio",
    "bbox_aspect_ratio",
    "geometry_compactness",
    "estimated_units_missing",
    "GEOID_missing",
]
MASK_GEOMETRY_COLS = [
    "mask_area_px",
    "mask_fill_ratio",
    "bbox_w_px",
    "bbox_h_px",
    "bbox_aspect_ratio",
    "geometry_compactness",
]
SUPPORTED_IMAGE_BACKBONES = ["efficientnet_b0", "efficientnet_v2_s", "convnext_tiny", "resnet50"]
SUPPORTED_STAGE2A_INPUT_MODES = [
    "rgb_only",
    "rgb_mask",
    "rgb_mask_geometry",
    "rgb_mask_ring",
    "rgb_mask_ring_geometry",
    "geometry_only",
]
SUPPORTED_STAGE2A_POOLING_MODES = ["global", "mask_m", "mask_m_ring"]


def class_names_from_config(config: Mapping[str, object] | None = None) -> List[str]:
    """Return checkpoint class names, falling back to the legacy Stage-2a order."""
    cfg = config or {}
    raw = cfg.get("class_names") or cfg.get("active_class_names")
    if isinstance(raw, str) and raw.strip():
        parsed = parse_list(raw, CLASS_NAMES)
        if parsed:
            return parsed
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        parsed = [str(x).strip() for x in raw if str(x).strip()]
        if parsed:
            return parsed
    num_classes = safe_int(cfg.get("num_classes"), len(CLASS_NAMES))
    if 0 < num_classes <= len(CLASS_NAMES):
        return list(CLASS_NAMES[:num_classes])
    return list(CLASS_NAMES)


def prob_columns_for_class_names(class_names: Sequence[str] | None = None) -> List[str]:
    return [f"prob_{name}" for name in list(class_names or CLASS_NAMES)]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv_rows(path: Path | str, limit: int = 0) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def write_csv_rows(path: Path | str, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_jsonable(obj: object) -> object:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def save_json(path: Path | str, obj: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, sort_keys=True)


def load_json(path: Path | str) -> object:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_list(value: str | Sequence[str] | None, default: Optional[Sequence[str]] = None) -> List[str]:
    if value is None or value == "":
        return list(default or [])
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [str(x).strip() for x in value if str(x).strip()]


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def has_value(value: object) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return False
    return True


def resolve_path(value: str, base_dir: Path | str | None = None) -> Path:
    p = Path(value)
    if p.is_absolute() or base_dir is None:
        return p
    return Path(base_dir) / p


def _load_mask_array(mask_path: Path | str) -> np.ndarray:
    arr = np.asarray(Image.open(mask_path).convert("L"))
    return (arr > 0).astype(np.uint8)


def mask_geometry_from_array(mask: np.ndarray) -> Dict[str, float]:
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    h, w = mask.shape
    area = int(mask.sum())
    fill = float(area / max(1, h * w))
    if area <= 0:
        return {
            "mask_area_px": 0.0,
            "mask_fill_ratio": 0.0,
            "bbox_w_px": 0.0,
            "bbox_h_px": 0.0,
            "bbox_aspect_ratio": 0.0,
            "geometry_compactness": 0.0,
        }
    ys, xs = np.where(mask > 0)
    bbox_w = int(xs.max() - xs.min() + 1)
    bbox_h = int(ys.max() - ys.min() + 1)
    bbox_area = max(1, bbox_w * bbox_h)
    aspect = float(bbox_w / max(1, bbox_h))
    compactness = float(area / bbox_area)
    return {
        "mask_area_px": float(area),
        "mask_fill_ratio": fill,
        "bbox_w_px": float(bbox_w),
        "bbox_h_px": float(bbox_h),
        "bbox_aspect_ratio": aspect,
        "geometry_compactness": compactness,
    }


def extract_geometry_features(mask: np.ndarray | Path | str | None, row: Mapping[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if mask is not None:
        try:
            arr = _load_mask_array(mask) if isinstance(mask, (str, Path)) else np.asarray(mask)
            out.update(mask_geometry_from_array((arr > 0).astype(np.uint8)))
        except Exception:
            out.update(mask_geometry_from_array(np.zeros((1, 1), dtype=np.uint8)))
    else:
        for key in ["mask_area_px", "mask_fill_ratio", "bbox_w_px", "bbox_h_px", "bbox_aspect_ratio", "geometry_compactness"]:
            out[key] = safe_float(row.get(key), 0.0)

    footprint_raw = row.get("footprint_m2")
    log_footprint_raw = row.get("log_footprint_m2")
    footprint = safe_float(footprint_raw, 0.0)
    if not has_value(footprint_raw) and has_value(log_footprint_raw):
        log_footprint = max(0.0, safe_float(log_footprint_raw, 0.0))
        footprint = float(np.expm1(log_footprint))
    else:
        log_footprint = math.log1p(max(0.0, footprint))
    out["footprint_m2"] = footprint
    out["log_footprint_m2"] = log_footprint
    units_raw = row.get("estimated_units", "")
    geoid_raw = row.get("GEOID", "")
    out["estimated_units"] = safe_float(units_raw, 0.0)
    out["estimated_units_missing"] = 1.0 if units_raw in ("", None) else 0.0
    out["GEOID_missing"] = 1.0 if geoid_raw in ("", None) else 0.0
    return out


def missing_geometry_columns(row: Mapping[str, object], geometry_cols: Sequence[str] | None = None) -> List[str]:
    return [c for c in list(geometry_cols or DEFAULT_GEOMETRY_COLS) if not has_value(row.get(c))]


def needs_mask_geometry(row: Mapping[str, object], geometry_cols: Sequence[str] | None = None) -> bool:
    requested = set(geometry_cols or DEFAULT_GEOMETRY_COLS)
    return any(c in requested and not has_value(row.get(c)) for c in MASK_GEOMETRY_COLS)


def geometry_vector(
    row: Mapping[str, object],
    geometry_cols: Sequence[str] | None = None,
    mask: np.ndarray | Path | str | None = None,
) -> np.ndarray:
    cols = list(geometry_cols or DEFAULT_GEOMETRY_COLS)
    feats = extract_geometry_features(mask, row)
    return np.asarray([safe_float(feats.get(c, row.get(c, 0.0)), 0.0) for c in cols], dtype=np.float32)


def apply_population_blend(
    neural_pop_log: np.ndarray,
    geometry: np.ndarray,
    blend_model: Mapping[str, object],
    source_values: Sequence[object] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a saved geometry-ridge population blend on log1p population."""
    neural = np.asarray(neural_pop_log, dtype=np.float64)
    geom = np.asarray(geometry, dtype=np.float64)
    if "models" in blend_model:
        return _apply_source_aware_population_blend(neural, geom, blend_model, source_values)
    return _apply_single_population_blend(neural, geom, blend_model)


def _apply_single_population_blend(
    neural: np.ndarray,
    geom: np.ndarray,
    blend_model: Mapping[str, object],
) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(blend_model["feature_mean"], dtype=np.float64).reshape(1, -1)
    sd = np.asarray(blend_model["feature_std"], dtype=np.float64).reshape(1, -1)
    beta = np.asarray(blend_model["beta"], dtype=np.float64)
    if geom.shape[1] != mu.shape[1]:
        raise ValueError(f"Blend geometry dim mismatch: got {geom.shape[1]}, expected {mu.shape[1]}")
    sd = np.where(np.abs(sd) < 1e-6, 1.0, sd)
    xs = (geom - mu) / sd
    xs = np.concatenate([np.ones((xs.shape[0], 1)), xs], axis=1)
    geometry_log = xs @ beta
    weight_neural = safe_float((blend_model.get("blend") or {}).get("weight_neural"), 1.0)
    blended_log = weight_neural * neural + (1.0 - weight_neural) * geometry_log
    return blended_log.astype(np.float64), geometry_log.astype(np.float64)


def _source_key(value: object) -> str:
    text = str(value or "").strip()
    return text if text else "__missing__"


def _select_source_population_model(blend_model: Mapping[str, object], source_value: object) -> Tuple[Mapping[str, object], str, bool]:
    models = blend_model.get("models") or {}
    if not isinstance(models, Mapping):
        raise ValueError("Source-aware population blend artifact must contain a mapping under 'models'.")
    source_key = _source_key(source_value)
    if source_key in models:
        return models[source_key], source_key, False
    fallback = blend_model.get("fallback_model") or blend_model.get("global_model")
    if isinstance(fallback, Mapping):
        return fallback, "__fallback__", True
    if "feature_mean" in blend_model and "feature_std" in blend_model and "beta" in blend_model:
        return blend_model, "__global__", True
    raise KeyError(
        f"No source-aware population blend model for source={source_key!r} and no fallback_model/global_model is available."
    )


def _apply_source_aware_population_blend(
    neural: np.ndarray,
    geom: np.ndarray,
    blend_model: Mapping[str, object],
    source_values: Sequence[object] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(neural) != len(geom):
        raise ValueError(f"Blend batch length mismatch: neural={len(neural)}, geometry={len(geom)}")
    sources = list(source_values or [""] * len(neural))
    if len(sources) != len(neural):
        raise ValueError(f"Blend source length mismatch: sources={len(sources)}, neural={len(neural)}")
    blended = np.zeros(len(neural), dtype=np.float64)
    geometry_log = np.zeros(len(neural), dtype=np.float64)
    for i, source in enumerate(sources):
        model, _, _ = _select_source_population_model(blend_model, source)
        pred_i, geom_i = _apply_single_population_blend(neural[i : i + 1], geom[i : i + 1], model)
        blended[i] = float(pred_i[0])
        geometry_log[i] = float(geom_i[0])
    return blended, geometry_log


def population_blend_metadata(
    blend_model: Mapping[str, object],
    source_values: Sequence[object] | None = None,
    n_rows: int | None = None,
) -> List[Dict[str, object]]:
    """Return per-row blend metadata for CSV/debug output."""
    if "models" not in blend_model:
        blend = blend_model.get("blend") or {}
        n = int(n_rows if n_rows is not None else len(source_values or []))
        return [
            {
                "population_blend_source": "__global__",
                "population_blend_input_source": "",
                "population_blend_used_fallback": False,
                "population_blend_weight_neural": safe_float(blend.get("weight_neural"), 1.0),
                "population_blend_weight_geometry": safe_float(blend.get("weight_geometry"), 0.0),
            }
            for _ in range(n)
        ]
    sources = list(source_values or [""] * int(n_rows or 0))
    out: List[Dict[str, object]] = []
    for source in sources:
        model, selected_source, used_fallback = _select_source_population_model(blend_model, source)
        blend = model.get("blend") or {}
        out.append(
            {
                "population_blend_source": selected_source,
                "population_blend_input_source": _source_key(source),
                "population_blend_used_fallback": bool(used_fallback),
                "population_blend_weight_neural": safe_float(blend.get("weight_neural"), 1.0),
                "population_blend_weight_geometry": safe_float(blend.get("weight_geometry"), 0.0),
            }
        )
    return out


def attach_geometry_columns(rows: Iterable[MutableMapping[str, object]], base_dir: Path | str | None = None) -> List[MutableMapping[str, object]]:
    out = []
    for row in rows:
        mask_path = str(row.get("mask_path") or row.get("mask_M") or "")
        geom = extract_geometry_features(resolve_path(mask_path, base_dir) if mask_path else None, row)
        for key, value in geom.items():
            row[key] = value
        out.append(row)
    return out


def effective_split_group(row: Mapping[str, object], group_col: str, row_idx: int) -> str:
    """Return the exact group value used for blocked splitting."""
    return str(row.get(group_col) or row.get("tile_id") or f"__row_{row_idx}")


def tile_train_val_test_split(
    rows: Sequence[Mapping[str, object]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    tile_col: str = "tile_base",
) -> Dict[str, List[int]]:
    if not rows:
        raise ValueError("Cannot split empty rows")
    group_to_idx: Dict[str, List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        group = effective_split_group(row, tile_col, i)
        group_to_idx[group].append(i)
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
    train_groups = set(groups[n_test + n_val :])

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


def split_manifest(rows: Sequence[Mapping[str, object]], split: Mapping[str, Sequence[int]], group_col: str) -> Dict[str, object]:
    return {
        "group_col": group_col,
        "counts": {k: len(v) for k, v in split.items()},
        "groups": {k: sorted({effective_split_group(rows[i], group_col, i) for i in v}) for k, v in split.items()},
        "raw_missing_group_counts": {k: sum(1 for i in v if not str(rows[i].get(group_col, ""))) for k, v in split.items()},
        "effective_group_fallback": f"{group_col} -> tile_id -> __row_i",
        "indices": {k: list(map(int, v)) for k, v in split.items()},
    }


def split_has_leakage(rows: Sequence[Mapping[str, object]], split: Mapping[str, Sequence[int]], group_col: str) -> bool:
    groups = {k: {effective_split_group(rows[i], group_col, i) for i in idxs} for k, idxs in split.items()}
    keys = list(groups)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if groups[a] & groups[b]:
                return True
    return False


class Stage2aDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Mapping[str, object]],
        indices: Optional[Sequence[int]] = None,
        img_size: int = 224,
        train: bool = False,
        input_mode: str = "rgb_mask",
        geometry_cols: Sequence[str] | None = None,
        crop_col: str = "crop_path",
        mask_col: str = "mask_path",
        ring_mask_col: str = "mask_R",
        ring_radius_px: int = 48,
        require_ring_mask: bool = True,
        id_col: str = "building_uid",
        base_dir: Path | str | None = None,
        aug_hflip: float = 0.5,
        aug_vflip: float = 0.0,
        aug_rot90: float = 0.0,
        seed: int = 42,
        require_targets: bool = True,
    ):
        self.rows = list(rows)
        self.indices = list(indices) if indices is not None else list(range(len(rows)))
        self.img_size = int(img_size)
        self.train = bool(train)
        self.input_mode = input_mode
        self.geometry_cols = list(geometry_cols or DEFAULT_GEOMETRY_COLS)
        self.crop_col = crop_col
        self.mask_col = mask_col
        self.ring_mask_col = ring_mask_col
        self.ring_radius_px = int(ring_radius_px)
        self.require_ring_mask = bool(require_ring_mask)
        self.id_col = id_col
        self.base_dir = Path(base_dir) if base_dir is not None else None
        self.aug_hflip = float(aug_hflip)
        self.aug_vflip = float(aug_vflip)
        self.aug_rot90 = float(aug_rot90)
        self.seed = int(seed)
        self.require_targets = require_targets
        self.rgb_tf = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.mask_tf = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.indices)

    def _path(self, row: Mapping[str, object], col: str) -> Path:
        value = str(row.get(col, ""))
        if not value:
            raise KeyError(f"Missing path column {col}")
        return resolve_path(value, self.base_dir)

    def _optional_path(self, row: Mapping[str, object], col: str) -> Optional[Path]:
        value = str(row.get(col, "") or "")
        if not value:
            return None
        path = resolve_path(value, self.base_dir)
        return path if path.exists() else None

    def _augment(
        self,
        rgb: Image.Image,
        mask: Image.Image,
        ring: Optional[Image.Image],
        local_idx: int,
    ) -> Tuple[Image.Image, Image.Image, Optional[Image.Image]]:
        if not self.train:
            return rgb, mask, ring
        rng = random.Random((self.seed * 1_000_003) + local_idx)
        if rng.random() < self.aug_hflip:
            rgb = TF.hflip(rgb)
            mask = TF.hflip(mask)
            if ring is not None:
                ring = TF.hflip(ring)
        if rng.random() < self.aug_vflip:
            rgb = TF.vflip(rgb)
            mask = TF.vflip(mask)
            if ring is not None:
                ring = TF.vflip(ring)
        if rng.random() < self.aug_rot90:
            k = rng.choice([1, 2, 3])
            angle = 90 * k
            rgb = TF.rotate(rgb, angle)
            mask = TF.rotate(mask, angle)
            if ring is not None:
                ring = TF.rotate(ring, angle)
        return rgb, mask, ring

    def _ring_tensor_from_mask(self, mask_t: torch.Tensor) -> torch.Tensor:
        radius = max(0, int(self.ring_radius_px))
        if radius <= 0:
            return torch.zeros_like(mask_t)
        kernel = radius * 2 + 1
        dilated = F.max_pool2d(mask_t.unsqueeze(0), kernel_size=kernel, stride=1, padding=radius).squeeze(0)
        return (dilated - mask_t).clamp(0.0, 1.0)

    def __getitem__(self, local_idx: int) -> Dict[str, object]:
        row_idx = self.indices[local_idx]
        row = self.rows[row_idx]
        crop_path = self._path(row, self.crop_col)
        mask_path = self._path(row, self.mask_col)
        needs_ring = self.require_ring_mask or self.input_mode in ("rgb_mask_ring", "rgb_mask_ring_geometry")
        ring_path = self._optional_path(row, self.ring_mask_col) if needs_ring else None
        rgb = Image.open(crop_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        ring = Image.open(ring_path).convert("L") if ring_path is not None else None
        rgb, mask, ring = self._augment(rgb, mask, ring, local_idx)
        rgb_t = self.rgb_tf(rgb)
        mask_t = self.mask_tf(mask).clamp(0.0, 1.0)
        if needs_ring:
            ring_t = self.mask_tf(ring).clamp(0.0, 1.0) if ring is not None else self._ring_tensor_from_mask(mask_t)
        else:
            ring_t = torch.zeros_like(mask_t)
        if self.input_mode == "rgb_only":
            x = rgb_t
        elif self.input_mode in ("rgb_mask_ring", "rgb_mask_ring_geometry"):
            x = torch.cat([rgb_t, mask_t, ring_t], dim=0)
        else:
            x = torch.cat([rgb_t, mask_t], dim=0)
        mask_for_geometry = np.asarray(mask) if needs_mask_geometry(row, self.geometry_cols) else None
        geom_feats = extract_geometry_features(mask_for_geometry, row)
        geom = torch.from_numpy(geometry_vector(row, self.geometry_cols, mask=mask_for_geometry))

        pop_raw = row.get("estimated_population")
        if not has_value(pop_raw):
            pop_raw = row.get("true_population")
        pop = safe_float(pop_raw, 0.0)
        type_raw = row.get("type_class")
        if not has_value(type_raw):
            type_raw = row.get("true_type_idx")
        type_name_raw = row.get("type_class_name")
        if not has_value(type_name_raw):
            type_name_raw = row.get("true_type_class", "")
        type_idx = safe_int(type_raw, CLASS_TO_IDX.get(str(type_name_raw), -1))
        if self.require_targets and type_idx < 0:
            raise ValueError(f"Missing type target for row {row_idx}")
        meta = {
            "building_uid": str(row.get(self.id_col, "")),
            "crop_path": str(row.get(self.crop_col, "")),
            "mask_path": str(row.get(self.mask_col, "")),
            "ring_mask_path": str(row.get(self.ring_mask_col, "")),
            "tile_base": str(row.get("tile_base", row.get("tile_id", ""))),
            "tile_id": str(row.get("tile_id", row.get("tile_base", ""))),
            "event_id": str(row.get("event_id", "")),
            "sam3_confidence": str(row.get("sam3_confidence", "")),
            "hazard_type": str(row.get("hazard_type", "")),
            "GEOID": str(row.get("GEOID", "")),
            "type_class_name": str(type_name_raw or ""),
            "classification_source": str(row.get("classification_source", "")),
            "estimated_population": pop,
            "has_population_target": bool(has_value(pop_raw)),
            "has_type_target": bool(type_idx >= 0),
        }
        for key, value in geom_feats.items():
            meta[key] = value
        return {
            "x": x,
            "mask": mask_t,
            "ring_mask": ring_t,
            "geometry": geom,
            "pop_log": torch.tensor(math.log1p(max(0.0, pop)), dtype=torch.float32),
            "type_idx": torch.tensor(type_idx, dtype=torch.long),
            "row_idx": torch.tensor(row_idx, dtype=torch.long),
            "meta": meta,
        }


def collate_stage2a(batch: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
        "ring_mask": torch.stack([b["ring_mask"] for b in batch], dim=0),
        "geometry": torch.stack([b["geometry"] for b in batch], dim=0),
        "pop_log": torch.stack([b["pop_log"] for b in batch], dim=0),
        "type_idx": torch.stack([b["type_idx"] for b in batch], dim=0),
        "row_idx": torch.stack([b["row_idx"] for b in batch], dim=0),
        "meta": [b["meta"] for b in batch],
    }


class Stage2aInferenceDataset(Stage2aDataset):
    def __init__(self, rows, img_size=224, crop_col="crop_path", mask_col="mask_path", id_col="building_uid", **kwargs):
        super().__init__(
            rows,
            indices=None,
            img_size=img_size,
            train=False,
            crop_col=crop_col,
            mask_col=mask_col,
            id_col=id_col,
            require_targets=False,
            **kwargs,
        )


def _default_weights(enum_name: str, pretrained: bool):
    if not pretrained:
        return None
    enum = getattr(models, enum_name, None)
    if enum is None:
        return None
    return getattr(enum, "DEFAULT", None)


def _adapt_conv_in_channels(old_conv: nn.Conv2d, input_channels: int) -> nn.Conv2d:
    if input_channels == old_conv.in_channels:
        return old_conv
    new_conv = nn.Conv2d(
        input_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )
    with torch.no_grad():
        new_conv.weight.zero_()
        copy_ch = min(input_channels, old_conv.in_channels)
        new_conv.weight[:, :copy_ch] = old_conv.weight[:, :copy_ch]
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    return new_conv


def build_image_feature_extractor(backbone_name: str, pretrained: bool, input_channels: int) -> Tuple[nn.Module, int]:
    name = str(backbone_name or "efficientnet_b0").lower()
    if name == "efficientnet_b0":
        backbone = models.efficientnet_b0(weights=_default_weights("EfficientNet_B0_Weights", pretrained))
        backbone.features[0][0] = _adapt_conv_in_channels(backbone.features[0][0], input_channels)
        return backbone.features, int(backbone.classifier[1].in_features)
    if name == "efficientnet_v2_s":
        if not hasattr(models, "efficientnet_v2_s"):
            raise ValueError("torchvision does not provide efficientnet_v2_s in this environment")
        backbone = models.efficientnet_v2_s(weights=_default_weights("EfficientNet_V2_S_Weights", pretrained))
        backbone.features[0][0] = _adapt_conv_in_channels(backbone.features[0][0], input_channels)
        return backbone.features, int(backbone.classifier[1].in_features)
    if name == "convnext_tiny":
        if not hasattr(models, "convnext_tiny"):
            raise ValueError("torchvision does not provide convnext_tiny in this environment")
        backbone = models.convnext_tiny(weights=_default_weights("ConvNeXt_Tiny_Weights", pretrained))
        backbone.features[0][0] = _adapt_conv_in_channels(backbone.features[0][0], input_channels)
        return backbone.features, int(backbone.classifier[-1].in_features)
    if name == "resnet50":
        backbone = models.resnet50(weights=_default_weights("ResNet50_Weights", pretrained))
        backbone.conv1 = _adapt_conv_in_channels(backbone.conv1, input_channels)
        features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        return features, int(backbone.fc.in_features)
    raise ValueError(f"Unsupported backbone_name={backbone_name}. Supported: {SUPPORTED_IMAGE_BACKBONES}")


def _downsample_mask(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return F.interpolate(mask, size=(h, w), mode="nearest")


def _masked_avg_pool(feat: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = _downsample_mask(mask, feat.shape[-2], feat.shape[-1])
    num = (feat * m).sum(dim=(2, 3))
    den = m.sum(dim=(2, 3)).clamp_min(eps)
    return num / den


class BuildingPopulationModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = False,
        input_channels: int = 4,
        backbone_name: str = "efficientnet_b0",
        geometry_dim: int = 0,
        geometry_hidden_dim: int = 64,
        type_conditioning_mode: str = "none",
        type_geometry_mode: str = "none",
        pooling_mode: str = "global",
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        if type_conditioning_mode not in ("none", "soft_probs", "detached_soft_probs", "logits", "hard_onehot"):
            raise ValueError(f"Unknown type_conditioning_mode={type_conditioning_mode}")
        if type_geometry_mode not in ("none", "concat"):
            raise ValueError(f"Unknown type_geometry_mode={type_geometry_mode}")
        if pooling_mode not in SUPPORTED_STAGE2A_POOLING_MODES:
            raise ValueError(f"Unknown pooling_mode={pooling_mode}")
        self.num_classes = int(num_classes)
        self.input_channels = int(input_channels)
        self.backbone_name = str(backbone_name or "efficientnet_b0")
        self.geometry_dim = int(geometry_dim)
        self.type_conditioning_mode = type_conditioning_mode
        self.type_geometry_mode = type_geometry_mode
        self.pooling_mode = pooling_mode

        self.features, feat_dim = build_image_feature_extractor(self.backbone_name, pretrained, input_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.shared_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim * (2 if pooling_mode == "mask_m_ring" else 1), hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.geometry_proj = None
        geom_out = 0
        if self.geometry_dim > 0:
            geom_out = int(geometry_hidden_dim)
            self.geometry_proj = nn.Sequential(
                nn.Linear(self.geometry_dim, geom_out),
                nn.ReLU(inplace=True),
                nn.LayerNorm(geom_out),
            )
        type_in_dim = hidden_dim + (geom_out if type_geometry_mode == "concat" else 0)
        self.type_head = nn.Sequential(
            nn.Linear(type_in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
        type_cond_dim = 0 if type_conditioning_mode == "none" else num_classes
        pop_in_dim = hidden_dim + geom_out + type_cond_dim
        self.pop_head = nn.Sequential(
            nn.Linear(pop_in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        geometry: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        ring_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ):
        z = self.features(x)
        if self.pooling_mode == "global":
            z = self.pool(z).flatten(1)
        elif self.pooling_mode == "mask_m":
            if mask is None:
                raise ValueError("pooling_mode='mask_m' requires mask tensor")
            z = _masked_avg_pool(z, mask)
        else:
            if mask is None or ring_mask is None:
                raise ValueError("pooling_mode='mask_m_ring' requires mask and ring_mask tensors")
            z_m = _masked_avg_pool(z, mask)
            z_r = _masked_avg_pool(z, ring_mask)
            z = torch.cat([z_m, z_r], dim=1)
        z = self.shared_fc(z)
        geom_z = None
        if self.geometry_proj is not None:
            if geometry is None:
                raise ValueError("geometry tensor is required for this model")
            geom_z = self.geometry_proj(geometry.float())
        type_parts = [z]
        if self.type_geometry_mode == "concat":
            if geom_z is None:
                raise ValueError("type_geometry_mode='concat' requires geometry features")
            type_parts.append(geom_z)
        type_logits = self.type_head(torch.cat(type_parts, dim=1))
        pop_parts = [z]
        if geom_z is not None:
            pop_parts.append(geom_z)
        if self.type_conditioning_mode != "none":
            if self.type_conditioning_mode == "soft_probs":
                t = torch.softmax(type_logits, dim=1)
            elif self.type_conditioning_mode == "detached_soft_probs":
                t = torch.softmax(type_logits, dim=1).detach()
            elif self.type_conditioning_mode == "hard_onehot":
                idx = torch.argmax(type_logits.detach(), dim=1)
                t = F.one_hot(idx, num_classes=self.num_classes).float()
            else:
                t = type_logits
            pop_parts.append(t)
        pop_log = self.pop_head(torch.cat(pop_parts, dim=1)).squeeze(-1)
        if return_dict:
            return {"pop_log": pop_log, "type_logits": type_logits, "type_probs": torch.softmax(type_logits, dim=1)}
        return pop_log, type_logits


class GeometryOnlyPopulationModel(nn.Module):
    """Small MLP baseline that uses polygon-derived geometry only."""

    def __init__(
        self,
        geometry_dim: int,
        num_classes: int = 5,
        hidden_dim: int = 128,
        type_conditioning_mode: str = "none",
        dropout: float = 0.1,
    ):
        super().__init__()
        if geometry_dim <= 0:
            raise ValueError("geometry_dim must be positive for geometry-only model")
        if type_conditioning_mode not in ("none", "soft_probs", "detached_soft_probs", "logits", "hard_onehot"):
            raise ValueError(f"Unknown type_conditioning_mode={type_conditioning_mode}")
        self.num_classes = int(num_classes)
        self.geometry_dim = int(geometry_dim)
        self.type_conditioning_mode = type_conditioning_mode
        self.encoder = nn.Sequential(
            nn.Linear(geometry_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.type_head = nn.Linear(hidden_dim, num_classes)
        type_cond_dim = 0 if type_conditioning_mode == "none" else num_classes
        self.pop_head = nn.Sequential(
            nn.Linear(hidden_dim + type_cond_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        geometry: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        ring_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ):
        if geometry is None:
            raise ValueError("geometry tensor is required for geometry-only model")
        z = self.encoder(geometry.float())
        type_logits = self.type_head(z)
        pop_parts = [z]
        if self.type_conditioning_mode != "none":
            if self.type_conditioning_mode == "soft_probs":
                t = torch.softmax(type_logits, dim=1)
            elif self.type_conditioning_mode == "detached_soft_probs":
                t = torch.softmax(type_logits, dim=1).detach()
            elif self.type_conditioning_mode == "hard_onehot":
                t = F.one_hot(torch.argmax(type_logits.detach(), dim=1), num_classes=self.num_classes).float()
            else:
                t = type_logits
            pop_parts.append(t)
        pop_log = self.pop_head(torch.cat(pop_parts, dim=1)).squeeze(-1)
        if return_dict:
            return {"pop_log": pop_log, "type_logits": type_logits, "type_probs": torch.softmax(type_logits, dim=1)}
        return pop_log, type_logits


def load_torch_checkpoint(path: Path | str, map_location: str | torch.device = "cpu") -> object:
    """Load a trusted local Stage-2a checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError:
        return torch.load(path, map_location=map_location, weights_only=False)


def load_state_dict_from_ckpt(path: Path | str) -> Dict[str, torch.Tensor]:
    ckpt = load_torch_checkpoint(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ["model_state", "state_dict", "model"]:
            value = ckpt.get(key)
            if isinstance(value, dict):
                return value
    return ckpt


def load_ckpt_config(path: Path | str) -> Dict[str, object]:
    ckpt = load_torch_checkpoint(path, map_location="cpu")
    if isinstance(ckpt, dict):
        cfg = ckpt.get("config") or ckpt.get("train_config") or {}
        return dict(cfg) if isinstance(cfg, dict) else {}
    return {}


def build_model_from_config(config: Mapping[str, object], pretrained: bool = False) -> BuildingPopulationModel:
    input_mode = str(config.get("input_mode", "rgb_mask"))
    geometry_cols = parse_list(config.get("geometry_cols"), DEFAULT_GEOMETRY_COLS)
    if input_mode == "geometry_only":
        return GeometryOnlyPopulationModel(
            geometry_dim=len(geometry_cols),
            num_classes=int(config.get("num_classes", len(CLASS_NAMES))),
            type_conditioning_mode=str(config.get("type_conditioning_mode", "none")),
            hidden_dim=int(config.get("hidden_dim", 128)),
            dropout=float(config.get("dropout", 0.1)),
        )
    use_geometry = "geometry" in input_mode or str(config.get("type_conditioning_mode", "none")) != "none" and bool(config.get("force_geometry", False))
    input_channels = 3 if input_mode == "rgb_only" else 5 if input_mode in ("rgb_mask_ring", "rgb_mask_ring_geometry") else 4
    return BuildingPopulationModel(
        num_classes=int(config.get("num_classes", len(CLASS_NAMES))),
        pretrained=pretrained,
        input_channels=input_channels,
        backbone_name=str(config.get("backbone_name", config.get("backbone", "efficientnet_b0"))),
        geometry_dim=len(geometry_cols) if use_geometry else 0,
        geometry_hidden_dim=int(config.get("geometry_hidden_dim", 64)),
        type_conditioning_mode=str(config.get("type_conditioning_mode", "none")),
        type_geometry_mode=str(config.get("type_geometry_mode", "none")),
        pooling_mode=str(config.get("pooling_mode", "global")),
        hidden_dim=int(config.get("hidden_dim", 512)),
        dropout=float(config.get("dropout", 0.2)),
    )


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < n_classes and 0 <= int(p) < n_classes:
            cm[int(t), int(p)] += 1
    return cm


def macro_f1_from_cm(cm: np.ndarray) -> Tuple[float, List[float]]:
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        den = 2 * tp + fp + fn
        f1s.append(float(2 * tp / den) if den > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0, f1s


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    if len(y_true) == 0:
        return 0.0
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if np.any(mask):
            ece += float(mask.mean() * abs(correct[mask].mean() - conf[mask].mean()))
    return ece


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int] | None = None,
    probs: Optional[np.ndarray] = None,
    class_names: Sequence[str] = CLASS_NAMES,
    ece_bins: int = 15,
) -> Dict[str, object]:
    y_true_np = np.asarray(y_true, dtype=np.int64)
    if probs is not None:
        probs = np.asarray(probs, dtype=np.float64)
        y_pred_np = probs.argmax(axis=1)
    elif y_pred is not None:
        y_pred_np = np.asarray(y_pred, dtype=np.int64)
    else:
        raise ValueError("Provide y_pred or probs")
    cm = confusion_matrix(y_true_np, y_pred_np, len(class_names))
    macro_f1, per_class = macro_f1_from_cm(cm)
    acc = float((y_true_np == y_pred_np).mean()) if len(y_true_np) else 0.0
    out: Dict[str, object] = {
        "type_accuracy": acc,
        "type_macro_f1": macro_f1,
        "type_per_class_f1": {name: per_class[i] for i, name in enumerate(class_names)},
        "confusion_matrix": cm.tolist(),
    }
    if probs is not None:
        clipped = np.clip(probs[np.arange(len(y_true_np)), y_true_np], 1e-12, 1.0)
        out["type_nll"] = float(-np.log(clipped).mean()) if len(clipped) else 0.0
        out["type_ece"] = expected_calibration_error(y_true_np, probs, bins=ece_bins)
    return out


def regression_metrics(
    y_true_log: Sequence[float],
    y_pred_log: Sequence[float],
    y_true: Optional[Sequence[float]] = None,
    y_pred: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    yt_log = np.asarray(y_true_log, dtype=np.float64)
    yp_log = np.asarray(y_pred_log, dtype=np.float64)
    yt = np.asarray(y_true, dtype=np.float64) if y_true is not None else np.expm1(yt_log)
    yp = np.asarray(y_pred, dtype=np.float64) if y_pred is not None else np.expm1(yp_log)
    yp = np.maximum(0.0, yp)
    mae = float(np.mean(np.abs(yp - yt))) if len(yt) else 0.0
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2))) if len(yt) else 0.0
    log_mae = float(np.mean(np.abs(yp_log - yt_log))) if len(yt_log) else 0.0
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    factor2 = float(np.mean(np.abs(yp_log - yt_log) <= math.log(2.0))) if len(yt_log) else 0.0
    return {
        "population_mae": mae,
        "population_rmse": rmse,
        "population_r2": float(r2),
        "population_log_mae": log_mae,
        "population_factor2_hit_rate": factor2,
    }


def population_bins(values: Sequence[float]) -> List[str]:
    out = []
    for v in values:
        x = float(v)
        if x <= 0:
            out.append("zero")
        elif x <= 100:
            out.append("lt_100")
        elif x <= 300:
            out.append("100_300")
        elif x <= 1000:
            out.append("300_1000")
        else:
            out.append("gt_1000")
    return out


def grouped_regression_metrics(rows: Sequence[Mapping[str, object]], group_col: str) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_col, ""))].append(row)
    out: Dict[str, Dict[str, float]] = {}
    for group, items in groups.items():
        yt = [safe_float(x.get("true_population", x.get("estimated_population")), 0.0) for x in items]
        yp = [safe_float(x.get("pred_population"), 0.0) for x in items]
        out[group or "__missing__"] = regression_metrics(np.log1p(yt), np.log1p(yp), yt, yp)
    return out


def prediction_rows_from_outputs(
    metas: Sequence[Mapping[str, object]],
    pop_log: np.ndarray,
    probs: np.ndarray,
    y_pop_log: Optional[np.ndarray] = None,
    y_type: Optional[np.ndarray] = None,
    class_names: Sequence[str] = CLASS_NAMES,
) -> List[Dict[str, object]]:
    rows = []
    names = list(class_names)
    prob_columns = prob_columns_for_class_names(names)
    pred_pop = np.maximum(0.0, np.expm1(pop_log))
    pred_idx = probs.argmax(axis=1)
    pred_conf = probs[np.arange(len(pred_idx)), pred_idx]
    for i, meta in enumerate(metas):
        pred_name = names[int(pred_idx[i])] if 0 <= int(pred_idx[i]) < len(names) else ""
        row: Dict[str, object] = {
            "building_uid": meta.get("building_uid", ""),
            "pred_population": float(pred_pop[i]),
            "pred_log1p_population": float(pop_log[i]),
            "pred_type_idx": int(pred_idx[i]),
            "pred_type_class": pred_name,
            "pred_type_conf": float(pred_conf[i]),
            "crop_path": meta.get("crop_path", ""),
            "mask_path": meta.get("mask_path", ""),
            "tile_base": meta.get("tile_base", ""),
            "GEOID": meta.get("GEOID", ""),
            "classification_source": meta.get("classification_source", ""),
        }
        for j, col in enumerate(prob_columns):
            row[col] = float(probs[i, j])
        if y_pop_log is not None:
            row["true_log1p_population"] = float(y_pop_log[i])
            row["true_population"] = float(np.expm1(y_pop_log[i]))
        if y_type is not None:
            row["true_type_idx"] = int(y_type[i])
            row["true_type_class"] = names[int(y_type[i])] if 0 <= int(y_type[i]) < len(names) else ""
        rows.append(row)
    return rows


def stage2a_prediction_fields(include_truth: bool = True, class_names: Sequence[str] = CLASS_NAMES) -> List[str]:
    fields = [
        "building_uid",
        "pred_population",
        "pred_log1p_population",
        "pred_type_idx",
        "pred_type_class",
        "pred_type_conf",
        *prob_columns_for_class_names(class_names),
        "crop_path",
        "mask_path",
        "tile_base",
        "GEOID",
        "classification_source",
    ]
    if include_truth:
        fields.extend(["true_population", "true_log1p_population", "true_type_idx", "true_type_class"])
    return fields


def metrics_from_prediction_rows(rows: Sequence[Mapping[str, object]], class_names: Sequence[str] = CLASS_NAMES) -> Dict[str, object]:
    y_true_type = [safe_int(r.get("true_type_idx"), -1) for r in rows]
    has_type = all(x >= 0 for x in y_true_type)
    y_true_pop = [safe_float(r.get("true_population"), 0.0) for r in rows]
    y_pred_pop = [safe_float(r.get("pred_population"), 0.0) for r in rows]
    metrics: Dict[str, object] = {}
    if has_type and rows:
        prob_columns = prob_columns_for_class_names(class_names)
        probs = np.asarray([[safe_float(r.get(col), 0.0) for col in prob_columns] for r in rows], dtype=np.float64)
        metrics.update(classification_metrics(y_true_type, probs=probs, class_names=class_names))
    metrics.update(regression_metrics(np.log1p(y_true_pop), np.log1p(y_pred_pop), y_true_pop, y_pred_pop))
    metrics["population_by_type"] = grouped_regression_metrics(rows, "true_type_class")
    metrics["population_by_classification_source"] = grouped_regression_metrics(rows, "classification_source")
    bin_rows = []
    for row, bin_name in zip(rows, population_bins(y_true_pop)):
        r2 = dict(row)
        r2["population_bin"] = bin_name
        bin_rows.append(r2)
    metrics["population_by_bin"] = grouped_regression_metrics(bin_rows, "population_bin")
    return metrics


def class_weights(rows: Sequence[Mapping[str, object]], indices: Sequence[int], alpha: float = 0.5, cap: float = 10.0) -> List[float]:
    labels = [safe_int(rows[i].get("type_class"), -1) for i in indices]
    counts = Counter(x for x in labels if x >= 0)
    if not counts:
        return [1.0 for _ in indices]
    max_count = max(counts.values())
    weights_by_class = {
        c: min(float(cap), (max_count / max(1, n)) ** float(alpha))
        for c, n in counts.items()
    }
    return [weights_by_class.get(x, 1.0) for x in labels]
