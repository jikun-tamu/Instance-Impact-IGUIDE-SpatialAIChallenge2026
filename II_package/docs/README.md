# Instance Impact Framework

This repository follows an instance-centric disaster impact workflow built around a shared building instance unit.

Detailed Stage-2b experiment logs and updates are maintained in `Stage2b.md`.

## Component-Level Summary

### Stage 1 - Building Instance Extraction

**Input**
- Pre-disaster satellite image
- Post-disaster satellite image

**Process**
- SAM3 segmentation to detect building instances
- Convert masks to building polygons
- Assign segmentation confidence
- Generate standardized per-building subimages (pre + post)

**Output**
- `instance_id`
- building polygon
- segmentation confidence
- pre-disaster subimage
- post-disaster subimage

Purpose: define the shared building instance unit used by all downstream models.

### Stage 2a - Occupancy Unit Estimation

**Input**
- pre-disaster subimage
- building polygon

**Process**
- model trained on satellite + demographic data
- infer residential occupancy units per building

**Output**
- predicted occupancy units

Purpose: estimate exposure capacity of each building.

### Stage 2b - Damage State Prediction

**Input**
- pre-disaster subimage
- post-disaster subimage
- building polygon

**Process**
- change-based damage inference model
- context-aware pooling using building mask + surrounding ring
- calibrated probability outputs
- ensemble uncertainty estimation

**Output**
- damage class (0-3)
- expected severity
- calibrated confidence metrics
- ensemble uncertainty metrics

Purpose: estimate physical disaster damage per building.

### Stage 3 - Impact Synthesis

**Input**
- Stage 2a: occupancy units
- Stage 2b: damage state

**Process**
- align predictions by `instance_id`
- combine exposure and damage

**Output**
- building-instance disaster impact representation

Example outputs:
- `{occupancy_units, damage_state}`
- derived impact score or impact category

Purpose: produce instance-level disaster impact assessment.

## Overall Pipeline

```text
Pre/Post Satellite Images
          |
          v
Stage 1 - Building Instance Extraction
          |
          v
Per-building subimages + polygons
          |
          +-------------------+
          |                   |
          v                   v
Stage 2a                   Stage 2b
Occupancy                  Damage State
Estimation                 Prediction
          \                 /
           \               /
            +-- Impact Synthesis --+
                         |
                         v
         Building-Instance Disaster Impact
```

## Core Design Principle

The framework is instance-centric:

```text
building instance
     |
     v
{occupancy exposure, damage severity}
     |
     v
instance-level disaster impact
```

## Current Integration Status

The framework now supports an end-to-end **driver-style** run on a single pre/post tile:

- Stage 1 (`SAM3`) -> instance polygons + `sam3_confidence`
- shared crop/mask generation (`256`, `M` + `R`) for downstream reuse
- Stage 2a inference -> `pred_population`, `pred_type_class`, `pred_type_conf`
- Stage 2b ensemble inference -> damage class + calibrated uncertainty metrics
- synthesis/presentation merge -> one instance-level table with joined confidences

Primary scripts used:

- `scripts/run_instance_impact_driver.py` (orchestrates Stage1 -> Stage2 -> presentation)
- `scripts/present_instance_results.py` (merges Stage1/2a/2b outputs and prints terminal summaries)

Notes:

- This current layer is a **results driver/presenter**, not a new modeling stage.
- In-domain behavior is better for flood-domain checkpoints on flood tiles.

## Instance-Level Visualization

Current visualization is based on Stage2b overlays and now supports adding Stage1/2a metrics:

- base image: `post_crop`
- region overlay: filled `mask_M` polygon (semi-transparent, no boundary stroke)
- color map: predicted Stage2b class (`0..3` green -> red)

Panel layout:

- **Top-left**: Stage2b metrics (`pred`, `gt`, `expected severity`, `pmax`, `margin`, `entropy`)
- **Bottom-left**: Stage1 confidence (`s1_conf`)
- **Bottom-right**: Stage2a summary (`pop`, `type`, `s2a_conf`)

Script:

- `scripts/visualize_stage2_overlays.py`

Key inputs:

- `--pred_jsonl` Stage2b ensemble output JSONL
- `--csv` shared/stage2 sample CSV with `post_crop` + `mask_M`
- `--stage2a_csv` optional Stage2a prediction CSV for bottom-right panel

## Execution Progress Log

This section records framework integration progress and runnable commands used for end-to-end checks.

### Milestone A: End-to-End Driver Assembled

Implemented orchestrator:

- `scripts/run_instance_impact_driver.py`

Current driver flow:

1. Stage 1 SAM3 inference on one pre/post pair
2. Shared subimage generation (`shared_instance_samples.csv`, `crops_pre/post`, `mask_M/R`)
3. Stage2a inference
4. Stage2b ensemble inference (with optional calibration)
5. Presentation merge (`scripts/present_instance_results.py`)
6. Instance-level overlays (`scripts/visualize_stage2_overlays.py`)

---

### Milestone B: Working Full Runs

#### 1) Flood-domain destroyed-heavy tile (in-domain check)

Tile:

- `nepal-flooding_00000442`

Command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_instance_impact_driver.py \
  --pre_image xBD/tier3/images/nepal-flooding_00000442_pre_disaster.png \
  --post_image xBD/tier3/images/nepal-flooding_00000442_post_disaster.png \
  --run_id e2e_nepal_flooding_00000442_destroyed_tiled_notebook \
  --overwrite_run_dir \
  --stage1_backend transformers \
  --stage1_device cuda:0 \
  --stage1_output_style notebook \
  --stage1_tile_size 512 \
  --stage1_overlap 64 \
  --stage1_min_size 30 \
  --stage1_batch_size 1 \
  --stage2a_ckpt stage2a/best_model.pt \
  --stage2b_ckpts "outputs/inference0.7273.pt,outputs/inference0.7066_seed9999.pt,outputs/inference0.7034_seed7777.pt" \
  --stage2b_configs "outputs/sweeps/flood_seed2025_weighted_gs24/run019_s2025_lr5e-5__wd0p05__bs16_ep20_cbtrue_rot0p25__cj0p03_/train_config.json,outputs/sweeps/flood_run019_seed_sweep5_manual/seed_9999/train_config.json,outputs/sweeps/flood_run019_seed_sweep5_manual/seed_7777/train_config.json" \
  --stage2b_calibration_dirs "outputs/calibration_run019_r48,outputs/calibration_seed9999_r48,outputs/calibration_seed7777_r48" \
  --stage2b_calibration_method temperature
```

Observed:

- Driver completed end-to-end.
- Stage2b predicted severe classes on this flood tile (major/destroyed behavior present).
- Presentation and visualization outputs were generated.

Output root:

- `outputs/driver_runs/e2e_nepal_flooding_00000442_destroyed_tiled_notebook`

---

#### 2) Flood-domain minor+major tile

Tile:

- `nepal-flooding_00000408` (xBD labels include both minor and major)

Command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_instance_impact_driver.py \
  --pre_image xBD/tier3/images/nepal-flooding_00000408_pre_disaster.png \
  --post_image xBD/tier3/images/nepal-flooding_00000408_post_disaster.png \
  --run_id e2e_nepal_flooding_00000408_minor_major \
  --overwrite_run_dir \
  --stage1_backend transformers \
  --stage1_device cuda:0 \
  --stage1_output_style notebook \
  --stage1_tile_size 512 \
  --stage1_overlap 64 \
  --stage1_min_size 30 \
  --stage1_batch_size 1 \
  --stage2a_ckpt stage2a/best_model.pt \
  --stage2b_ckpts "outputs/inference0.7273.pt,outputs/inference0.7066_seed9999.pt,outputs/inference0.7034_seed7777.pt" \
  --stage2b_configs "outputs/sweeps/flood_seed2025_weighted_gs24/run019_s2025_lr5e-5__wd0p05__bs16_ep20_cbtrue_rot0p25__cj0p03_/train_config.json,outputs/sweeps/flood_run019_seed_sweep5_manual/seed_9999/train_config.json,outputs/sweeps/flood_run019_seed_sweep5_manual/seed_7777/train_config.json" \
  --stage2b_calibration_dirs "outputs/calibration_run019_r48,outputs/calibration_seed9999_r48,outputs/calibration_seed7777_r48" \
  --stage2b_calibration_method temperature
```

Output root:

- `outputs/driver_runs/e2e_nepal_flooding_00000408_minor_major`

Generated artifacts include:

- `stage1/`
- `shared_instances_r48/shared_instance_samples.csv`
- `stage2a_predictions.csv`
- `stage2b_ensemble.jsonl`
- `instance_results_presented.csv`
- `instance_results_top_uncertain.csv`
- `vis_instance_level/`

---

### Milestone C: Presentation Layer Enhancements

Presentation merge script:

- `scripts/present_instance_results.py`

Current terminal summary includes:

- Stage1/2a/2b join coverage
- counts by Stage2a type
- counts by Stage2b damage class (all rows + valid-only)
- quantiles for Stage1 confidence, Stage2a outputs, Stage2b uncertainty metrics
- top uncertain instances

Visualization overlay script:

- `scripts/visualize_stage2_overlays.py`

Current panel layout:

- top-left: Stage2b metrics
- bottom-left: Stage1 `s1_conf`
- bottom-right: Stage2a `pop`, `type`, `s2a_conf`

---

### Known Limitation (Current)

- `generate_shared_instance_subimages.py` currently skips `MULTIPOLYGON` WKT rows (`bad_wkt`), which can slightly reduce instance count in some runs.
