# Stage-2a Integration Notes

## Scope

This file records Stage-2a integration status in the Instance Impact Framework, with emphasis on:

- what the provided `stage2a` implementation does today,
- what was sanity-checked against Stage-1 shared artifacts,
- and what CLI scripts are now added for framework wiring.

Current objective is **end-to-end framework assembly first**, not Stage-2a model optimization.

---

## What We Confirmed About Original Stage-2a

The provided `stage2a` implementation has two distinct pieces:

1. A rule-based geospatial pipeline (`stage2a/code.ipynb`) that:
   - classifies building type from parcel/land-use + footprint area,
   - estimates units from type (fixed for duplex/triplex; area-based for multi-family),
   - estimates population via:
     - `estimated_population = estimated_units * people_per_unit_ratio * occupancy_rate`,
   - applies default ratios and caps, and reports quality flags.

2. A multi-task vision model (`stage2a/building_population_model.ipynb`) that predicts:
   - building type class logits,
   - log1p(population) regression.

Important clarification:

- The original pipeline does **not** infer units by reversing model-predicted population.
- The intended chain in rule logic is:
  - `type -> units -> population`.

---

## Sanity Run on Stage-1 Shared Assets

Shared artifacts used:

- `outputs/shared_instances_sanity_2tiles_v2_r48/shared_instance_samples.csv`
- per-instance `pre_crop` and `mask_M`

Quick adapter + inference run produced:

- input rows: 114
- output rows: 114
- output file:
  - `outputs/shared_instances_sanity_2tiles_v2_r48/stage2a_sanity_predictions.csv`

Observed behavior:

- Pipeline execution and output schema looked correct.
- Predicted classes/population values were plausible for an out-of-domain sanity pass, but not yet quality-tuned for deployment.

Decision:

- For downstream Stage-3 synthesis at this phase, use `pred_population` as primary exposure signal.
- Keep `pred_type_class` (and units if later available) as auxiliary diagnostic fields.

---

## New CLI Scripts Added

### 1) `scripts/build_stage2a_infer_csv.py`

Purpose:

- Convert Stage-1 shared instance table into Stage-2a inference input table.

Input:

- `shared_instance_samples.csv`

Output columns:

- `building_uid`
- `crop_path` (mapped from `pre_crop` by default)
- `mask_path` (mapped from `mask_M` by default)
- optional passthrough metadata (`tile_id`, `event_id`, `sam3_confidence`, `hazard_type` when available)

Notes:

- Validates required columns.
- Verifies crop/mask file existence by default.
- Supports `--skip_missing_paths` for permissive runs.

---

### 2) `scripts/infer_stage2a.py`

Purpose:

- Run standalone Stage-2a model inference on per-instance crop + mask assets.

Model contract:

- EfficientNet-B0 backbone, 4-channel input (RGB + mask), multi-task heads.
- Mirrors architecture from `stage2a/building_population_model.ipynb`.

Input:

- CSV with `building_uid`, `crop_path`, `mask_path` (plus optional metadata).
- checkpoint path via `--ckpt`.

Output:

- per-instance predictions including:
  - `pred_population`
  - `pred_log1p_population`
  - `pred_type_idx`
  - `pred_type_class`
  - `pred_type_conf`
  - full class probabilities
  - passthrough metadata fields

Terminal summaries:

- class histogram
- population quantiles
- example predictions

---

## Example Commands

Build Stage-2a inference CSV from shared Stage-1 artifacts:

```bash
python3 scripts/build_stage2a_infer_csv.py \
  --shared_csv outputs/shared_instances_sanity_2tiles_v2_r48/shared_instance_samples.csv \
  --out_csv outputs/shared_instances_sanity_2tiles_v2_r48/stage2a_infer_input.csv \
  --crop_col pre_crop \
  --mask_col mask_M
```

Run Stage-2a inference:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/infer_stage2a.py \
  --input_csv outputs/shared_instances_sanity_2tiles_v2_r48/stage2a_infer_input.csv \
  --ckpt <PATH_TO_STAGE2A_CHECKPOINT.pt> \
  --out_csv outputs/shared_instances_sanity_2tiles_v2_r48/stage2a_sanity_predictions.csv \
  --batch_size 64 \
  --num_workers 4 \
  --print_examples 10
```

---

## Current Status

- Stage-2a now has CLI scripts for direct integration with Stage-1 shared artifacts.
- Framework-level wiring path is in place for:
  - Stage-1 shared instance assets -> Stage-2a predictions -> Stage-3 synthesis input.
- Next recommended step is implementing Stage-3 synthesis script and schema lock, while keeping Stage-2a model improvement for later iterations.
