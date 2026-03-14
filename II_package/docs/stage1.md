# Stage 1 Implementation Log

This document records what we implemented for Stage 1, why changes were made, and the exact commands used during the sanity workflow on xBD.

## Scope and Goal

Stage 1 target behavior in this project:
- Input: pre/post disaster imagery (xBD tiles).
- Process:
  - run SAM3 on pre-disaster images,
  - generate instance masks and confidence scores,
  - convert masks to per-instance polygons,
  - export per-image JSON listing building instances.
- Output:
  - `masks/*.tif` (instance labels),
  - `masks/*_scores.tif` (confidence scores),
  - `labels/*_prediction.json` (instance-level polygon + confidence),
  - optional visual annotation artifacts.

This Stage 1 output is now used as upstream input for shared subimage generation (for Stage 2a/2b compatibility).

## Final Sanity Result (2 tiles)

Sanity run output root:
- `outputs/stage1_sanity_2tiles_v2`

Observed summary:
- images: `2`
- instances: `124`
- per-image instances: `46`, `78`
- no skipped images

Instance JSON validation:
- `outputs/stage1_sanity_2tiles_v2/labels/joplin-tornado_00000000_pre_disaster_prediction.json` -> `features.xy = 46`
- `outputs/stage1_sanity_2tiles_v2/labels/joplin-tornado_00000001_pre_disaster_prediction.json` -> `features.xy = 78`

This confirms Stage 1 now emits instance-level JSON (not a collapsed single polygon).

## Commands We Ran

### 1) Environment and GPU sanity

```bash
python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("visible_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    try:
        torch.cuda.set_device(i)
        _ = torch.empty(1, device=f"cuda:{i}")
        print(f"gpu{i}: OK")
    except Exception as e:
        print(f"gpu{i}: FAIL -> {e}")
PY
```

### 2) Stage 1 sanity inference (final working command)

```bash
python3 stage1/SAM3_Final_20260226/scripts/run_sam3_building_infer.py \
  --input xBD/tier3/images \
  --output outputs/stage1_sanity_2tiles_v2 \
  --pattern "*_pre_disaster.png" \
  --max-images 2 \
  --prompt building \
  --min-size 100 \
  --output-style notebook \
  --device cuda:0 \
  --backend transformers
```

### 3) Verify JSON instance counts

```bash
python3 - <<'PY'
import json, glob
for p in sorted(glob.glob("outputs/stage1_sanity_2tiles_v2/labels/*_prediction.json")):
    d = json.load(open(p))
    print(p, "features_xy=", len(d["features"]["xy"]))
PY
```

### 4) Generate shared Stage-2-compatible assets from Stage 1 JSON

```bash
python3 scripts/generate_shared_instance_subimages.py \
  --stage1_labels_dir outputs/stage1_sanity_2tiles_v2/labels \
  --pre_images_dir xBD/tier3/images \
  --post_images_dir xBD/tier3/images \
  --out_root outputs/shared_instances_sanity_2tiles_v2_r48 \
  --crop_size 256 \
  --ring_radius_px 48 \
  --strict_images \
  --num_workers 4 \
  --chunk_size 100 \
  --log_every 50
```

Observed shared generation summary:
- collected rows: `114`
- written rows: `114`
- `bad_wkt: 10` (these are `MULTIPOLYGON` entries currently skipped by parser)

## Key Issues Encountered and Resolved

### A) Package and backend setup issues

- `samgeo` install name confusion:
  - direct `pip install samgeo` failed on mirror.
  - resolved by using `segment-geospatial` package family.
- mixed SAM2/SAM3 dependencies caused resolver churn and backend mismatch.
- final stable mode used for this workflow: `--backend transformers` with working `SamGeo3` import.

### B) CUDA/HPC instability

- observed transient CUDA driver initialization failures on some allocations.
- mitigated by re-allocating interactive session and using single healthy GPU.
- final run used 1 GPU (`cuda:0`) and passed.

### C) Gated model access

- initially blocked with HF gated-repo errors.
- once access became available, model download and inference proceeded.

### D) Transformers backend batch API mismatch (code fix)

Problem:
- notebook mode attempted `set_image_batch()` which is not supported for transformers backend.

Fix:
- patched notebook pipeline to avoid batch API for `transformers` and use per-image path.

### E) Silent polygon generation failure (code fix)

Problem:
- vectorization errors were swallowed (`try/except: pass`), causing missing `labels/*_prediction.json`.

Fix:
- enforced fail-fast behavior:
  - polygon generation failure now raises explicit runtime error with image id.
  - `run_polygons=True` now requires scores/masks.

### F) Raster dtype incompatibility in polygonization (code fix)

Problem:
- label masks were `uint32`, rejected by `rasterio.features.shapes`.

Fix:
- changed label raster dtype to `int32` where generated/stiched in pipeline.

### G) Output directory creation regression (code fix)

Problem:
- `sam3.save_masks(...)` failed in new output dir because `masks/` was not pre-created.

Fix:
- ensured `labels/`, `masks/`, `annotations/` directories are created before processing.

### H) Instance collapse in JSON (code fix)

Problem:
- old vectorization path could output overly collapsed geometry (not instance-level).

Fix:
- rewrote notebook vectorization to:
  - iterate per instance label,
  - polygonize per-label mask components,
  - compute per-label confidence from score raster,
  - emit one JSON feature per instance.

## Current Stage 1 Code Adjustments

Modified files:
- `stage1/SAM3_Final_20260226/src/sam3_final/pipeline.py`
  - transformers-compatible notebook path
  - fail-fast polygon enforcement
  - int32 label dtype handling
  - required output directory creation
- `stage1/SAM3_Final_20260226/src/sam3_final/notebook_outputs.py`
  - per-label instance polygonization
  - per-instance confidence extraction
  - targeted filtering of noisy GDAL deprecation line

Added bridge/shared-generation utilities:
- `scripts/build_stage2_index_from_stage1.py`
- `scripts/generate_shared_instance_subimages.py`

## Notes and Rationale

- `annotations/` can be empty in this sanity run because full annotation image writing is controlled by `--full-annotation` (not enabled in the run command).
- `NotGeoreferencedWarning` is expected for PNG xBD tiles without geotransform.
- `bad_wkt` in shared generator currently corresponds to `MULTIPOLYGON` entries skipped by its parser; this is a known, acceptable simplification for now.

## Checklist Status

- [x] Stage 1 model runs on 2-tile sanity case.
- [x] Masks and score rasters are generated.
- [x] Per-image instance JSON files are generated.
- [x] JSON contains per-instance polygon + confidence records.
- [x] Shared Stage-2-compatible subimages generated from Stage 1 JSON.
- [ ] Optional: support `MULTIPOLYGON` in shared generator (currently skipped).
- [ ] Optional: enable and validate full annotation PNG generation with `--full-annotation`.

