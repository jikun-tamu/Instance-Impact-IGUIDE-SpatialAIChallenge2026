# Stage-2 Reinstantiation Record (Flood-Only)

## Executive Summary
- Stage status: Stage-2b (damage state prediction) is rebuilt end-to-end from raw xBD data with a flood-only focus.
- Data pipeline: index -> crop/mask generation -> audits is stable, reproducible, and anchored on pre-disaster polygons (`pre_then_post`).
- Current default ("freeze"): `run019`-style setup with `ring_radius_px=48`, mild weighted sampling, and `mask_m_ring` pooling.
- Training posture: CLI-first PyTorch workflow, single-GPU baseline with DDP support, and HPC-oriented troubleshooting documented.
- Inference posture: top checkpoints copied for inference, calibration artifacts supported, 3-model weighted ensemble available with uncertainty metrics.
- Visualization posture: per-sample post-crop overlays implemented with filled polygon masks, top-left metric panel, and GT-label filtering.
- Role of this document: detailed Stage-2b logbook of commands, outcomes, ablations, and operational notes; high-level framework lives in `README.md`.

## Objective
Rebuild the Stage-2 building damage severity workflow from raw xBD data with a flood-only focus, CLI-first operation, and strict auditing.

The Stage-2 task assumes:
- input: paired pre/post building-centered subimages and a building polygon anchor
- output: per-building ordinal severity prediction and confidence/probabilities

All preparation and training artifacts are generated from raw xBD tiles/labels for reproducibility.

## Current Scope
- Hazard focus: **flooding only**
- Baseline prep folder naming: `outputs/flood_stage2_prep_r48`
- Baseline model config frozen from best run: `run019` (weighted sampler + `mask_m_ring`)
- Execution style: user runs commands in HPC; this README records known-good commands and outcomes

## Current Scripts
- `scripts/build_stage2_index.py`
- `scripts/preprocess_stage2_crops.py`
- `scripts/audit_stage2_samples.py`

## Standard Commands (Recorded)
Run from repository root:

```bash
python3 scripts/build_stage2_index.py \
  --root xBD/tier3 \
  --out outputs/stage2_index_flood.csv \
  --hazards flooding \
  --require_images
```

```bash
python3 scripts/preprocess_stage2_crops.py \
  --index_csv outputs/stage2_index_flood.csv \
  --out_root outputs/flood_stage2_prep_r48 \
  --crop_size 256 \
  --ring_radius_px 48 \
  --polygon_source pre_then_post \
  --dilate_backend auto \
  --png_compress_level 1 \
  --num_workers 16 \
  --chunk_size 300 \
  --log_every 200
```

```bash
python3 scripts/audit_stage2_samples.py \
  --csv outputs/flood_stage2_prep_r48/stage2_samples.csv \
  --sample_masks 2000
```

## Expected Outputs (Quick Checkpoints)
Use these as run-health checkpoints. Exact numbers can differ by split/version, but format should match.

### 1) `build_stage2_index.py`
Expected tail output pattern:

```text
Wrote CSV: outputs/stage2_index_flood.csv
Rows (buildings): <N_rows>
Tiles retained: <N_tiles>
Tiles by hazard:
  flooding: <N_flood_tiles>
Damage class counts:
  class_0: <count>
  class_1: <count>
  class_2: <count>
  class_3: <count>
Skipped diagnostics:
  empty_tile_no_features: <count>
  missing_or_unknown_subtype: <count>
```

What to verify:
- `Wrote CSV` path is correct
- `Rows (buildings)` is non-zero
- all classes are present (class_1/class_2 may be small)

### 2) `preprocess_stage2_crops.py`
Expected output pattern:

```text
Loaded rows: <N_rows> from outputs/stage2_index_flood.csv
Config: ... num_workers=16 chunk_size=300 ...
dilate_backend_resolved <scipy|opencv|python>
Prepared chunks: <N_chunks>
progress <done>/<total> (<pct>%) | written <count> | skip_img <count> | skip_poly <count> | skip_open <count> | skip_exist <count> | elapsed_s <sec>
...
Wrote: outputs/flood_stage2_prep_r48/stage2_samples.csv
skipped_existing <count>
skipped_missing_images <count>
skipped_missing_polygon <count>
skipped_open_error <count>
total_rows <N_total>
written_rows <N_written>
```

What to verify:
- backend resolves to `scipy` or `opencv` for speed
- `written_rows` equals `total_rows` for full-coverage CSV (existing artifacts are reused and still indexed)
- skip counters stay low (ideally zero except `skipped_existing` on reruns)

Behavior note:
- On reruns without `--overwrite`, `skipped_existing` can be high, and this is expected.
- Even when skipped_existing is high, rows are still written to `stage2_samples.csv` by referencing existing artifact paths.

### 3) `audit_stage2_samples.py`
Expected output pattern:

```text
CSV: outputs/flood_stage2_prep_r48/stage2_samples.csv
rows: <N_rows>
damage_class counts:
  0 <count>
  1 <count>
  2 <count>
  3 <count>
missing output files:
  pre_crop 0
  post_crop 0
  mask_M 0
  mask_R 0
mask area sample stats (first <K> rows):
  M area min/mean/max: <min> <mean> <max>
  M zero-area masks: 0
  R area min/mean/max: <min> <mean> <max>
  R zero-area masks: 0
```

What to verify:
- all missing-file counters are `0`
- zero-area mask counts are `0`
- mask area values are non-trivial (not all tiny or all saturated)

## Environment Note
Recommended minimum packages for current preprocessing and audits:

```bash
conda install -y numpy pillow scipy
```

## Flood-Only Stage-2 Prep + Baseline (Natural Sampler)

Baseline naming note:
- `outputs/flood_stage2_prep_r48` is the canonical baseline name going forward.
- It is equivalent to the earlier `outputs/flood_stage2_prep_full` dataset (built with `--ring_radius_px 48`).

### Flood prep audit snapshot

- Rows: `41,582`
- Class counts:
  - `c0=31,225`
  - `c1=5,134`
  - `c2=4,721`
  - `c3=502`
- File integrity: missing `pre_crop/post_crop/mask_M/mask_R` all `0`
- Mask integrity: zero-area `M` and `R` both `0` (sampled 2,000 rows)

Train/val split counts (`seed=42`, tile split):
- train (`34,971`): `{0: 26712, 1: 4134, 2: 3681, 3: 444}`
- val (`6,611`): `{0: 4513, 1: 1000, 2: 1040, 3: 58}`

### Flood baseline command used

```bash
SEEDS="42" \
NPROC_PER_NODE="4" \
CSV_PATH="outputs/flood_stage2_prep_r48/stage2_samples.csv" \
LRS="5e-5" \
WDS="0.1" \
BATCH_SIZES="12" \
EPOCHS_LIST="20" \
WARMUP_EPOCHS_LIST="1" \
LR_SCHEDULERS="cosine" \
NUM_WORKERS="4" \
CLASS_BALANCE_LIST="false" \
AUG_HFLIP_LIST="0.5" \
AUG_VFLIP_LIST="0.0" \
AUG_ROT90_LIST="0.25" \
AUG_COLOR_JITTER_LIST="0.03" \
EXTRA_ARGS="--sampler_mode natural --pretrained --early_stop_patience 5 --best_metric macro_f1 --best_tiebreak_metric qwk --ema_decay 0.999 --coral_label_smoothing 0.02 --event_metrics --save_val_predictions --change_fusion pre_post_diff --diff_abs_scale 1.0 --pooling_mode mask_m --print_per_class_f1 --print_confusion_matrix" \
OUT_ROOT="outputs/sweeps/flood_baseline_natural_maskm" \
bash scripts/run_sweep_stage2.sh
```

### Flood baseline result (single run)

| run | best_epoch | best_macro_f1 | best_qwk | best_ece | best_val_loss | best_c0_f1 | best_c1_f1 | best_c2_f1 | best_c3_f1 | final_epoch | final_macro_f1 | final_qwk | final_ece |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| run001 | 8 | 0.6038 | 0.6924 | 0.0987 | 0.23978 | 0.8946 | 0.4084 | 0.6652 | 0.4471 | 13 | 0.5978 | 0.6900 | 0.1133 |

Interpretation:
- Flood-only setup yields stable minority severity learning (`c1/c2`) with meaningful separation.
- `c1/c2` are meaningfully learnable (`~0.41` and `~0.67` at best checkpoint).
- This run establishes a strong baseline for flood-focused development.

## Flood 5-Seed (mask_m) Interim Highlight

While running the flood 5-seed sweep (`mask_m`) with GPU 1 excluded on node `h014`, one strong run was:

- Run: `run004_s2025_lr5e-5__wd0p1__bs16_ep20_cbfalse_rot0p25__cj0p03_`
- Best checkpoint:
  - `epoch=4`
  - `macro_f1=0.6759`
  - `qwk=0.7007`
  - `ece=0.0521`
  - `val_loss=0.19582`
  - `per_class_f1=[0.9193, 0.4594, 0.6584, 0.6667]`

Notes:
- This is materially stronger than earlier flood seed-42 baseline and confirms improved stability/performance under flood-only setup.
- Keep this recorded as an interim benchmark while waiting for full 5-seed completion and `mask_m` vs `mask_m_ring` comparison.

## Flood 5-Seed Sweep (mask_m only, 3-GPU on healthy devices 0/2/3)

Exact command used:

```bash
PYTHON_BIN="$(which python3)" \
CUDA_VISIBLE_DEVICES=0,2,3 \
SEEDS="42,123,3407,2025,7777" \
NPROC_PER_NODE="3" \
CSV_PATH="outputs/flood_stage2_prep_r48/stage2_samples.csv" \
LRS="5e-5" \
WDS="0.1" \
BATCH_SIZES="16" \
EPOCHS_LIST="20" \
WARMUP_EPOCHS_LIST="1" \
LR_SCHEDULERS="cosine" \
NUM_WORKERS="12" \
CLASS_BALANCE_LIST="false" \
AUG_HFLIP_LIST="0.5" \
AUG_VFLIP_LIST="0.0" \
AUG_ROT90_LIST="0.25" \
AUG_COLOR_JITTER_LIST="0.03" \
EXTRA_ARGS="--sampler_mode natural --pretrained --early_stop_patience 5 --best_metric macro_f1 --best_tiebreak_metric qwk --ema_decay 0.999 --coral_label_smoothing 0.02 --event_metrics --save_val_predictions --change_fusion pre_post_diff --diff_abs_scale 1.0 --pooling_mode mask_m --print_per_class_f1 --print_confusion_matrix" \
OUT_ROOT="outputs/sweeps/flood_5seed_maskm_cv023" \
bash scripts/run_sweep_stage2.sh
```

Sweep summary:

| run | best_epoch | best_macro_f1 | best_qwk | best_ece | best_val_loss | best_c0_f1 | best_c1_f1 | best_c2_f1 | best_c3_f1 | final_epoch | final_macro_f1 | final_qwk | final_ece |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| run001 (seed=42) | 8 | 0.6064 | 0.6936 | 0.0961 | 0.24206 | 0.8957 | 0.4435 | 0.6689 | 0.4176 | 13 | 0.5960 | 0.6857 | 0.1145 |
| run002 (seed=123) | 4 | 0.6198 | 0.7261 | 0.0601 | 0.18753 | 0.9401 | 0.5184 | 0.6676 | 0.3529 | 9 | 0.6125 | 0.7172 | 0.0560 |
| run003 (seed=3407) | 4 | 0.6179 | 0.6894 | 0.0613 | 0.19846 | 0.9252 | 0.4535 | 0.6210 | 0.4719 | 9 | 0.6082 | 0.6830 | 0.0737 |
| run004 (seed=2025) | 4 | 0.6759 | 0.7007 | 0.0521 | 0.19582 | 0.9193 | 0.4594 | 0.6584 | 0.6667 | 9 | 0.6746 | 0.7005 | 0.0728 |
| run005 (seed=7777) | 5 | 0.6588 | 0.7592 | 0.0623 | 0.18580 | 0.9309 | 0.4763 | 0.7028 | 0.5254 | 10 | 0.6437 | 0.7527 | 0.0749 |

Interpretation:
- Flood + `mask_m` is consistently strong across seeds with meaningful `c1/c2` recovery.
- Best macro-F1 reached **0.6759** (seed 2025), with very competitive per-class profile.
- `c3` is more variable by seed than `c1/c2`.
- This is the reference setting for `mask_m` vs `mask_m_ring` comparison.

## Flood 5-Seed Sweep (mask_m_ring, 3-GPU on healthy devices 0/2/3)

Exact command used:

```bash
PYTHON_BIN="$(which python3)" \
CUDA_VISIBLE_DEVICES=0,2,3 \
SEEDS="42,123,3407,2025,7777" \
NPROC_PER_NODE="3" \
CSV_PATH="outputs/flood_stage2_prep_r48/stage2_samples.csv" \
LRS="5e-5" \
WDS="0.1" \
BATCH_SIZES="16" \
EPOCHS_LIST="20" \
WARMUP_EPOCHS_LIST="1" \
LR_SCHEDULERS="cosine" \
NUM_WORKERS="12" \
CLASS_BALANCE_LIST="false" \
AUG_HFLIP_LIST="0.5" \
AUG_VFLIP_LIST="0.0" \
AUG_ROT90_LIST="0.25" \
AUG_COLOR_JITTER_LIST="0.03" \
EXTRA_ARGS="--sampler_mode natural --pretrained --early_stop_patience 5 --best_metric macro_f1 --best_tiebreak_metric qwk --ema_decay 0.999 --coral_label_smoothing 0.02 --event_metrics --save_val_predictions --change_fusion pre_post_diff --diff_abs_scale 1.0 --pooling_mode mask_m_ring --print_per_class_f1 --print_confusion_matrix" \
OUT_ROOT="outputs/sweeps/flood_5seed_maskmring_cv023" \
bash scripts/run_sweep_stage2.sh
```

Sweep summary:

| run | best_epoch | best_macro_f1 | best_qwk | best_ece | best_val_loss | best_c0_f1 | best_c1_f1 | best_c2_f1 | best_c3_f1 | final_epoch | final_macro_f1 | final_qwk | final_ece |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| run001 (seed=42) | 3 | 0.6633 | 0.8203 | 0.0376 | 0.16549 | 0.9222 | 0.4755 | 0.7501 | 0.5053 | 8 | 0.6603 | 0.8133 | 0.1103 |
| run002 (seed=123) | 5 | 0.6695 | 0.8466 | 0.0425 | 0.12308 | 0.9529 | 0.5104 | 0.7473 | 0.4673 | 10 | 0.6615 | 0.8414 | 0.0719 |
| run003 (seed=3407) | 4 | 0.6487 | 0.8048 | 0.0403 | 0.13360 | 0.9399 | 0.4980 | 0.6827 | 0.4742 | 9 | 0.6447 | 0.7994 | 0.0844 |
| run004 (seed=2025) | 5 | 0.7144 | 0.8242 | 0.0628 | 0.14898 | 0.9359 | 0.4881 | 0.7558 | 0.6777 | 10 | 0.6986 | 0.8121 | 0.0966 |
| run005 (seed=7777) | 3 | 0.6944 | 0.8584 | 0.0340 | 0.12195 | 0.9488 | 0.4899 | 0.7608 | 0.5781 | 8 | 0.6917 | 0.8552 | 0.0774 |

Interpretation:
- `mask_m_ring` outperformed `mask_m` in this 5-seed flood setting on macro-F1 and QWK.
- `c2` and `c3` both improved notably in many seeds, suggesting surrounding-context signal is helpful for flood severity discrimination.
- Current best single run in this comparison: **0.7144 macro-F1** (seed 2025, `mask_m_ring`).

## Flood 5-Seed Sweep (mask_m_ring + mild weighted sampler, 3-GPU 0/2/3)

Exact command used:

```bash
PYTHON_BIN="$(which python3)" \
CUDA_VISIBLE_DEVICES=0,2,3 \
SEEDS="42,123,3407,2025,7777" \
NPROC_PER_NODE="3" \
CSV_PATH="outputs/flood_stage2_prep_r48/stage2_samples.csv" \
LRS="5e-5" \
WDS="0.1" \
BATCH_SIZES="16" \
EPOCHS_LIST="20" \
WARMUP_EPOCHS_LIST="1" \
LR_SCHEDULERS="cosine" \
NUM_WORKERS="12" \
CLASS_BALANCE_LIST="true" \
CLASS_BALANCE_ALPHA_LIST="0.2" \
CLASS_BALANCE_CAP_LIST="3.0" \
AUG_HFLIP_LIST="0.5" \
AUG_VFLIP_LIST="0.0" \
AUG_ROT90_LIST="0.25" \
AUG_COLOR_JITTER_LIST="0.03" \
EXTRA_ARGS="--sampler_mode weighted --pretrained --early_stop_patience 5 --best_metric macro_f1 --best_tiebreak_metric qwk --ema_decay 0.999 --coral_label_smoothing 0.02 --event_metrics --save_val_predictions --change_fusion pre_post_diff --diff_abs_scale 1.0 --pooling_mode mask_m_ring --print_per_class_f1 --print_confusion_matrix" \
OUT_ROOT="outputs/sweeps/flood_5seed_maskmring_weighted_mild_cv023" \
bash scripts/run_sweep_stage2.sh
```

Sweep summary:

| run | best_epoch | best_macro_f1 | best_qwk | best_ece | best_val_loss | best_c0_f1 | best_c1_f1 | best_c2_f1 | best_c3_f1 | final_epoch | final_macro_f1 | final_qwk | final_ece |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| run001 (seed=42) | 3 | 0.6695 | 0.8188 | 0.0520 | 0.17094 | 0.9176 | 0.4813 | 0.7589 | 0.5200 | 8 | 0.6542 | 0.8123 | 0.1127 |
| run002 (seed=123) | 8 | 0.6744 | 0.8374 | 0.0689 | 0.14444 | 0.9513 | 0.5123 | 0.7340 | 0.5000 | 13 | 0.6582 | 0.8285 | 0.0808 |
| run003 (seed=3407) | 2 | 0.6584 | 0.7925 | 0.0330 | 0.13854 | 0.9368 | 0.5108 | 0.6667 | 0.5192 | 7 | 0.6450 | 0.8056 | 0.0796 |
| run004 (seed=2025) | 5 | 0.7266 | 0.8227 | 0.0723 | 0.15221 | 0.9341 | 0.4807 | 0.7496 | 0.7419 | 10 | 0.7203 | 0.8190 | 0.0915 |
| run005 (seed=7777) | 15 | 0.7051 | 0.8607 | 0.0845 | 0.16165 | 0.9475 | 0.4844 | 0.7673 | 0.6212 | 20 | 0.6986 | 0.8605 | 0.0846 |

Interpretation:
- Mild weighted sampling with `mask_m_ring` is competitive and slightly improves peak macro-F1 in several seeds.
- Best single run increased to **0.7266 macro-F1** (seed 2025), with strong `c2/c3` behavior.
- This setting is a strong candidate for follow-up confirmatory runs and calibration checks.

## HPC Troubleshooting Notes (Anvil)

This section records failure modes seen during flood multi-seed sweeps and the recovery playbook.

### 1) Python startup failure (`can't find encoding`)

Symptom:
- Python fails before script start with:
  - `Fatal Python error: init_fs_encoding`
  - `LookupError: no codec search functions registered: can't find encoding`

Cause observed:
- Conda env stdlib files were partially missing/inconsistent (`encodings` package contents missing, `python310.zip` missing).

In-place repair sequence (preferred first):

```bash
module purge --force
module load anaconda/2024.02-py311
source /apps/anvil/external/apps/conda/2024.02/etc/profile.d/conda.sh
unset PYTHONHOME PYTHONPATH

conda install -y -p /anvil/scratch/x-jliu7/conda-envs/projii --force-reinstall python=3.10
conda install -y -p /anvil/scratch/x-jliu7/conda-envs/projii --force-reinstall pip setuptools wheel

conda activate /anvil/scratch/x-jliu7/conda-envs/projii
python3 -c "import encodings,sys; print(sys.executable); print('ok')"
```

If this fails:
- Start a fresh interactive node/session and retry.
- Rebuild env only if repair + new session both fail.

### 2) DDP launch fails with `CUDA error: out of memory` at `torch.cuda.set_device`

Symptom:
- Failure occurs before model allocation, often on `local_rank=1`.

Interpretation:
- Usually node/GPU-state issue (bad GPU context), not true model memory OOM.

Per-GPU health probe:

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

Observed on node `h014`:
- `gpu0/2/3: OK`
- `gpu1: FAIL -> CUDA error: out of memory`

Workaround:
- Exclude bad GPU via `CUDA_VISIBLE_DEVICES`, then match `NPROC_PER_NODE`.

Example:

```bash
CUDA_VISIBLE_DEVICES=0,2,3 NPROC_PER_NODE=3 ...
```

Fallback:
- `NPROC_PER_NODE=1` to keep experiments running if node remains unstable.

### 3) Sweep launcher interpreter robustness

`scripts/run_sweep_stage2.sh` now supports:
- `PYTHON_BIN` (defaults to `python3`)

Use explicit interpreter from active env:

```bash
PYTHON_BIN="$(which python3)" bash scripts/run_sweep_stage2.sh
```

## Flood Weighted GS Attempt (Seed 2025, 24 runs)

Goal:
- Run a bounded exploratory grid search around the new default:
  - `pooling_mode=mask_m_ring`
  - mild weighted sampler
  - fixed seed `2025`
- Cap total runs at 24.

Exact command used:

```bash
PYTHON_BIN="$(which python3)" \
CUDA_VISIBLE_DEVICES=0,2,3 \
SEEDS="2025" \
NPROC_PER_NODE="3" \
CSV_PATH="outputs/flood_stage2_prep_r48/stage2_samples.csv" \
LRS="3e-5,5e-5,7e-5" \
WDS="0.05,0.1" \
BATCH_SIZES="16,24" \
EPOCHS_LIST="20" \
WARMUP_EPOCHS_LIST="1" \
LR_SCHEDULERS="cosine" \
NUM_WORKERS="12" \
CLASS_BALANCE_LIST="true" \
CLASS_BALANCE_ALPHA_LIST="0.1,0.2" \
CLASS_BALANCE_CAP_LIST="2.0,3.0" \
AUG_HFLIP_LIST="0.5" \
AUG_VFLIP_LIST="0.0" \
AUG_ROT90_LIST="0.25" \
AUG_COLOR_JITTER_LIST="0.03" \
EXTRA_ARGS="--sampler_mode weighted --pretrained --early_stop_patience 5 --best_metric macro_f1 --best_tiebreak_metric qwk --ema_decay 0.999 --coral_label_smoothing 0.02 --event_metrics --save_val_predictions --change_fusion pre_post_diff --diff_abs_scale 1.0 --pooling_mode mask_m_ring --print_per_class_f1 --print_confusion_matrix" \
OUT_ROOT="outputs/sweeps/flood_seed2025_weighted_gs24" \
bash scripts/run_sweep_stage2.sh
```

Grid size confirmation:
- `3 (LR) * 2 (WD) * 2 (batch size) * 2 (alpha) * 2 (cap) = 24 runs`

### Interim Stats from Completed Runs (27 `ok` runs before termination)

Mean and standard deviation across all completed runs (`N=27`) for reported numeric metrics:

| metric | mean | std | min | max |
|---|---:|---:|---:|---:|
| best_macro_f1 | 0.7196 | 0.0053 | 0.7058 | 0.7273 |
| best_qwk | 0.8198 | 0.0039 | 0.8117 | 0.8255 |
| best_ece | 0.0710 | 0.0132 | 0.0411 | 0.0921 |
| best_val_loss | 0.1560 | 0.0098 | 0.1408 | 0.1767 |
| best_c0_f1 | 0.9338 | 0.0013 | 0.9308 | 0.9354 |
| best_c1_f1 | 0.4873 | 0.0113 | 0.4636 | 0.5024 |
| best_c2_f1 | 0.7444 | 0.0086 | 0.7232 | 0.7588 |
| best_c3_f1 | 0.7128 | 0.0206 | 0.6721 | 0.7480 |
| final_macro_f1 | 0.7081 | 0.0054 | 0.6997 | 0.7184 |
| final_qwk | 0.8165 | 0.0041 | 0.8062 | 0.8210 |
| final_ece | 0.0952 | 0.0031 | 0.0899 | 0.1006 |

Top-3 completed runs by `best_macro_f1`:

| rank | run_name | best_macro_f1 | best_qwk | best_ece | best_val_loss | best_c0_f1 | best_c1_f1 | best_c2_f1 | best_c3_f1 | final_macro_f1 | final_qwk | final_ece |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | run019_s2025_lr5e-5__wd0p05__bs16_ep20_cbtrue_rot0p25__cj0p03_ | 0.7273 | 0.8255 | 0.0573 | 0.14701 | 0.9346 | 0.4994 | 0.7585 | 0.7167 | 0.7059 | 0.8209 | 0.0932 |
| 2 | run013_s2025_lr3e-5__wd0p1__bs24_ep20_cbtrue_rot0p25__cj0p03_ | 0.7254 | 0.8216 | 0.0864 | 0.17080 | 0.9354 | 0.4928 | 0.7341 | 0.7395 | 0.7148 | 0.8203 | 0.0951 |
| 3 | run014_s2025_lr3e-5__wd0p1__bs24_ep20_cbtrue_rot0p25__cj0p03_ | 0.7254 | 0.8216 | 0.0864 | 0.17080 | 0.9354 | 0.4928 | 0.7341 | 0.7395 | 0.7148 | 0.8203 | 0.0951 |

Selection note:
- Freeze `run019` config for downstream actions (calibration and confirmatory multi-seed):
  - `lr=5e-5`, `wd=0.05`, `batch_size=16`
  - weighted sampler (mild): `alpha=0.2`, `cap=3.0`
  - `pooling_mode=mask_m_ring`, `change_fusion=pre_post_diff`
- Rationale:
  - highest observed `best_macro_f1` in completed GS (`0.7273`)
  - strong `c2/c3` balance (`0.7585/0.7167`)
  - better calibration at best checkpoint than other top candidates (`best_ece=0.0573` vs `0.0864`)
- Secondary stable backup:
  - `run013/014` settings (`lr=3e-5`, `wd=0.1`, `bs=24`) with slightly lower peak macro-F1 but strong final-epoch retention.

## Ring Radius Ablation (r=24, seed 2025)

Goal:
- Evaluate `ring_radius_px=24` while holding the frozen `run019` training config constant.

Sweep command used:

```bash
PYTHON_BIN="$(which python3)" \
CUDA_VISIBLE_DEVICES=0,2,3 \
SEEDS="2025" \
NPROC_PER_NODE="3" \
CSV_PATH="outputs/flood_stage2_prep_r24/stage2_samples.csv" \
LRS="5e-5" \
WDS="0.05" \
BATCH_SIZES="16" \
EPOCHS_LIST="20" \
WARMUP_EPOCHS_LIST="1" \
LR_SCHEDULERS="cosine" \
NUM_WORKERS="12" \
CLASS_BALANCE_LIST="true" \
CLASS_BALANCE_ALPHA_LIST="0.2" \
CLASS_BALANCE_CAP_LIST="3.0" \
AUG_HFLIP_LIST="0.5" \
AUG_VFLIP_LIST="0.0" \
AUG_ROT90_LIST="0.25" \
AUG_COLOR_JITTER_LIST="0.03" \
EXTRA_ARGS="--sampler_mode weighted --pretrained --early_stop_patience 5 --best_metric macro_f1 --best_tiebreak_metric qwk --ema_decay 0.999 --coral_label_smoothing 0.02 --event_metrics --save_val_predictions --change_fusion pre_post_diff --diff_abs_scale 1.0 --pooling_mode mask_m_ring --print_per_class_f1 --print_confusion_matrix" \
OUT_ROOT="outputs/sweeps/ring_radius_ablation_run019_r24" \
bash scripts/run_sweep_stage2.sh
```

Final sweep summary:

| run_name | status | best_epoch | best_macro_f1 | best_qwk | best_ece | best_val_loss | best_c0_f1 | best_c1_f1 | best_c2_f1 | best_c3_f1 | final_epoch | final_macro_f1 | final_qwk | final_ece |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| run001_s2025_lr5e-5__wd0p05__bs16_ep20_cbtrue_a0p2__cap3p0__rot0p25__cj0p03_ | ok | 3 | 0.7245 | 0.8165 | 0.0409 | 0.14683 | 0.9345 | 0.5000 | 0.7377 | 0.7258 | 8 | 0.7087 | 0.8130 | 0.0895 |

Interpretation:
- `r=24` remains strong under the frozen weighted `run019` setup.
- Compared against the prior `r=48` best (`0.7273`), this `r=24` run is slightly lower in peak macro-F1 (`0.7245`) but very close.
- QWK is slightly lower than `r=48` best (`0.8165` vs `0.8255`), but still in the same high-performance band.
- Calibration and loss at best checkpoint are favorable (`best_ece=0.0409`, `best_val_loss=0.14683`), with ECE better and val loss essentially matched vs `r=48` (`0.0573`, `0.14701`).
- Class-wise profile is balanced: `c0=0.9345`, `c1=0.5000`, `c2=0.7377`, `c3=0.7258`; relative to `r=48`, `c1/c3` are slightly higher while `c2` is lower.
- Final-epoch retention is stable (`final_macro_f1=0.7087`, `final_qwk=0.8130`, `final_ece=0.0895`) and very close to `r=48` (`0.7059`, `0.8209`, `0.0932`).

## Ring Radius Ablation (r=72, seed 2025)

Goal:
- Evaluate `ring_radius_px=72` while holding the frozen `run019` training config constant.

Sweep command used:

```bash
PYTHON_BIN="$(which python3)" \
CUDA_VISIBLE_DEVICES=0,2,3 \
SEEDS="2025" \
NPROC_PER_NODE="3" \
CSV_PATH="outputs/flood_stage2_prep_r72/stage2_samples.csv" \
LRS="5e-5" \
WDS="0.05" \
BATCH_SIZES="16" \
EPOCHS_LIST="20" \
WARMUP_EPOCHS_LIST="1" \
LR_SCHEDULERS="cosine" \
NUM_WORKERS="12" \
CLASS_BALANCE_LIST="true" \
CLASS_BALANCE_ALPHA_LIST="0.2" \
CLASS_BALANCE_CAP_LIST="3.0" \
AUG_HFLIP_LIST="0.5" \
AUG_VFLIP_LIST="0.0" \
AUG_ROT90_LIST="0.25" \
AUG_COLOR_JITTER_LIST="0.03" \
EXTRA_ARGS="--sampler_mode weighted --pretrained --early_stop_patience 5 --best_metric macro_f1 --best_tiebreak_metric qwk --ema_decay 0.999 --coral_label_smoothing 0.02 --event_metrics --save_val_predictions --change_fusion pre_post_diff --diff_abs_scale 1.0 --pooling_mode mask_m_ring --print_per_class_f1 --print_confusion_matrix" \
OUT_ROOT="outputs/sweeps/ring_radius_ablation_run019_r72" \
bash scripts/run_sweep_stage2.sh
```

Final sweep summary:

| run_name | status | best_epoch | best_macro_f1 | best_qwk | best_ece | best_val_loss | best_c0_f1 | best_c1_f1 | best_c2_f1 | best_c3_f1 | final_epoch | final_macro_f1 | final_qwk | final_ece |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| run001_s2025_lr5e-5__wd0p05__bs16_ep20_cbtrue_a0p2__cap3p0__rot0p25__cj0p03_ | ok | 3 | 0.7186 | 0.8255 | 0.0387 | 0.14359 | 0.9363 | 0.4796 | 0.7569 | 0.7018 | 8 | 0.7116 | 0.8243 | 0.0876 |

Interpretation:
- `r=72` remains competitive but is below the `r=48` peak macro-F1 (`0.7186` vs `0.7273`) and below `r=24` (`0.7245`).
- QWK is strong (`0.8255`) and matches the best `r=48` QWK.
- Calibration/loss at best checkpoint are favorable (`best_ece=0.0387`, `best_val_loss=0.14359`), slightly better than both `r=48` and `r=24`.
- Per-class profile shifts toward higher `c2` (`0.7569`) with lower `c1`/`c3` (`0.4796`/`0.7018`) compared with `r=24`.
- Final-epoch retention is stable (`final_macro_f1=0.7116`, `final_qwk=0.8243`, `final_ece=0.0876`).

## Ring Generation Runtime (r=24 vs r=48 vs r=72)

Source:
- `scripts/preprocess_stage2_crops.py` loop over `ring_radius_px` with identical settings except radius.
- Terminal log recorded `elapsed_seconds` at completion for each radius.

| ring_radius_px | elapsed_seconds | elapsed_minutes | relative_to_r24 |
|---:|---:|---:|---:|
| 24 | 335.3 | 5.6 | 1.00x |
| 48 | 1946.6 | 32.4 | 5.80x |
| 72 | 7383.1 | 123.1 | 22.02x |

Quick takeaway:
- Runtime increases sharply with radius in this environment.
- `r=72` incurs a very large preprocessing cost for modest modeling gain, so `r=48` remains a strong default trade-off.

## Calibration Results (Frozen r=48 / run019)

Calibration command used:

```bash
python3 scripts/calibrate_stage2.py \
  --predictions_json outputs/sweeps/flood_seed2025_weighted_gs24/run019_s2025_lr5e-5__wd0p05__bs16_ep20_cbtrue_rot0p25__cj0p03_/val_predictions_best.json \
  --out_dir outputs/calibration_run019_r48 \
  --ece_bins 15 \
  --iters 300 \
  --lr 0.03 \
  --device cpu \
  --num_examples 12
```

Reported metrics:

| method | NLL | ECE |
|---|---:|---:|
| raw | 0.5471 | 0.0573 |
| temperature | 0.4880 | 0.0356 |
| vector exponents | 0.4200 | 0.0259 |

Learned calibration parameters:
- Scalar temperature: `T=1.5116`
- Vector exponents: `[0.652, 0.399, 0.612, 1.311]`

Artifacts written:
- `outputs/calibration_run019_r48/calibration_metrics.json`
- `outputs/calibration_run019_r48/calibrated_probs_temperature.npy`
- `outputs/calibration_run019_r48/calibrated_probs_vector.npy`
- `outputs/calibration_run019_r48/calibration_examples.json`
- `outputs/calibration_run019_r48/confidence_diagnostics.json`

Follow-up classification check on the same validation predictions:
- raw macro-F1: `0.7273`
- temperature macro-F1: `0.7273` (no argmax flips)
- vector macro-F1: `0.7207` (`99` argmax flips vs raw)

Takeaway:
- Calibration improved reliability metrics (NLL/ECE), especially for vector calibration.
- For classification decisions, temperature scaling is safer here (no F1 drop), while vector calibration can reduce macro-F1.

## Run019 Seed Sweep (5 seeds, manual)

Context:
- The original `run_sweep_stage2.sh` attempt hit DDP rendezvous port conflicts (`EADDRINUSE` on `29500`) in this HPC session.
- Runs were completed manually with explicit `--master_port` per seed under:
  - `outputs/sweeps/flood_run019_seed_sweep5_manual/seed_<seed>/`

Completed seeds and best-checkpoint metrics:

| seed | best_epoch | best_macro_f1 | best_qwk | best_ece | best_val_loss | best_c0_f1 | best_c1_f1 | best_c2_f1 | best_c3_f1 | final_macro_f1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2 | 0.6676 | 0.8149 | 0.0363 | 0.16935 | 0.9144 | 0.4772 | 0.7494 | 0.5294 | 0.6509 |
| 123 | 4 | 0.6669 | 0.8378 | 0.0548 | 0.13371 | 0.9494 | 0.5085 | 0.7413 | 0.4685 | 0.6528 |
| 3407 | 2 | 0.6600 | 0.8083 | 0.0340 | 0.13464 | 0.9370 | 0.5254 | 0.6870 | 0.4906 | 0.6359 |
| 7777 | 1 | 0.7034 | 0.8566 | 0.0403 | 0.13594 | 0.9363 | 0.4730 | 0.7647 | 0.6395 | 0.6858 |
| 9999 | 3 | 0.7066 | 0.7991 | 0.0803 | 0.19082 | 0.9033 | 0.5294 | 0.6939 | 0.7000 | 0.6897 |

Aggregate over 5 runs:
- `best_macro_f1`: mean `0.6809`, std `0.0199`
- `best_qwk`: mean `0.8233`, std `0.0210`
- `best_ece`: mean `0.0491`, std `0.0172`

Selection for ensemble diversity:
- Use new checkpoints from `seed=9999` and `seed=7777`.

## New Inference Checkpoints

Copy commands used:

```bash
cp outputs/sweeps/flood_run019_seed_sweep5_manual/seed_9999/stage2_best.pt outputs/inference0.7066_seed9999.pt
cp outputs/sweeps/flood_run019_seed_sweep5_manual/seed_7777/stage2_best.pt outputs/inference0.7034_seed7777.pt
```

Current 3-checkpoint ensemble set:
- `outputs/inference0.7273.pt` (frozen run019 anchor)
- `outputs/inference0.7066_seed9999.pt`
- `outputs/inference0.7034_seed7777.pt`

## Ensemble Uncertainty Inference (Top-3 checkpoints)

Script:
- `scripts/infer_stage2_ensemble.py`

Core outputs per sample:
- each model logits/probs
- weighted ensemble logits/probs (weights `4:3:2`)
- uncertainty metrics:
  - `var_predicted_class_prob_weighted`
  - `var_expected_severity_weighted`
  - `pmax`, `margin`, `entropy`

Smoke-test command (used):

```bash
python3 scripts/infer_stage2_ensemble.py \
  --csv outputs/flood_stage2_prep_r48/stage2_samples.csv \
  --ckpts "outputs/inference0.7273.pt,outputs/inference0.7066_seed9999.pt,outputs/inference0.7034_seed7777.pt" \
  --configs "outputs/sweeps/flood_seed2025_weighted_gs24/run019_s2025_lr5e-5__wd0p05__bs16_ep20_cbtrue_rot0p25__cj0p03_/train_config.json,outputs/sweeps/flood_run019_seed_sweep5_manual/seed_9999/train_config.json,outputs/sweeps/flood_run019_seed_sweep5_manual/seed_7777/train_config.json" \
  --weights "4,3,2" \
  --out_jsonl outputs/inference_ensemble_top3_new_r48.jsonl \
  --batch_size 64 \
  --num_workers 8 \
  --device cuda \
  --print_examples 5 \
  --log_every_steps 50 \
  --limit 200
```

Observed behavior:
- Model logits differ across all 3 checkpoints (ensemble diversity restored).
- Uncertainty metrics respond as expected (low on easy samples, higher on disagreement cases).
- `--limit 200` metrics are not used as final performance evidence; this run is for pipeline/uncertainty sanity.

## Calibration Results for New Two Models (Same Method as run019)

Completed with the same calibrator settings (`ece_bins=15`, `iters=300`, `lr=0.03`, `device=cpu`, `num_examples=12`):

```bash
python3 scripts/calibrate_stage2.py \
  --predictions_json outputs/sweeps/flood_run019_seed_sweep5_manual/seed_9999/val_predictions_best.json \
  --out_dir outputs/calibration_seed9999_r48 \
  --ece_bins 15 \
  --iters 300 \
  --lr 0.03 \
  --device cpu \
  --num_examples 12
```

Observed metrics (`outputs/calibration_seed9999_r48/calibration_metrics.json`):
- raw: `NLL=0.6055`, `ECE=0.0803`
- temperature: `NLL=0.5529`, `ECE=0.0471`, `T=1.4810`
- vector: `NLL=0.5009`, `ECE=0.0280`, `exp=[0.549, 0.583, 0.440, 1.268]`

```bash
python3 scripts/calibrate_stage2.py \
  --predictions_json outputs/sweeps/flood_run019_seed_sweep5_manual/seed_7777/val_predictions_best.json \
  --out_dir outputs/calibration_seed7777_r48 \
  --ece_bins 15 \
  --iters 300 \
  --lr 0.03 \
  --device cpu \
  --num_examples 12
```

Observed metrics (`outputs/calibration_seed7777_r48/calibration_metrics.json`):
- raw: `NLL=0.3847`, `ECE=0.0403`
- temperature: `NLL=0.3836`, `ECE=0.0396`, `T=1.0674`
- vector: `NLL=0.3557`, `ECE=0.0218`, `exp=[0.556, 1.020, 0.763, 1.339]`

Takeaway:
- Calibration completed successfully for both new models.
- Reliability (NLL/ECE) improved in both cases, with vector calibration giving the largest reductions.

## Final Calibrated Ensemble Uncertainty Summary (r48)

Run:
- `scripts/infer_stage2_ensemble.py` with:
  - checkpoints: `inference0.7273.pt`, `inference0.7066_seed9999.pt`, `inference0.7034_seed7777.pt`
  - weights: `4:3:2`
  - calibration method: `temperature`
  - calibration dirs: `calibration_run019_r48`, `calibration_seed9999_r48`, `calibration_seed7777_r48`

Reported uncertainty summary:
- `mean var_predicted_class_prob_weighted: 0.015591`
- `mean var_expected_severity_weighted: 0.021733`
- `mean pmax/margin/entropy: 0.8707 / 0.7767 / 0.4060`

Interpretation:
- Confidence is less over-peaked after temperature calibration (lower mean `pmax`, higher entropy).
- Disagreement-based uncertainty metrics are now tracked and available per sample in output JSONL.

Next step:
- Add per-sample prediction visualization to inspect logits/probabilities/uncertainty jointly with image context.

## Per-Sample Prediction Visualization (Implemented)

New script:
- `scripts/visualize_stage2_overlays.py`

What it renders:
- base image: `post_crop`
- polygon overlay: `mask_M` filled region (semi-transparent, default opacity `0.5`)
- class color palette (predicted class `0 -> 3`): green -> yellow-green -> orange -> red
- top-left multiline metrics panel:
  - `pred`, `gt`
  - `exp severity` with variance in parentheses
  - `pmax`, `margin`, `entropy`

Implemented controls:
- `--max_outputs` (custom number of outputs)
- `--gt_labels` filter (e.g. `2`, `3`, `2,3`)
- `--fill_opacity` (default `0.5`)
- output directory configurable via `--out_dir`

Alignment robustness update:
- Visualization now joins predictions to CSV rows using `sample_index` first (exact row alignment), then falls back to `(event_id, tile_id, bldg_uid)` key if needed.
- Dataset key uniqueness check on `flood_stage2_prep_r48` showed no duplicate identity keys (`41582` unique of `41582` rows).

Example commands used:

```bash
python3 scripts/visualize_stage2_overlays.py \
  --pred_jsonl outputs/inference_ensemble_top3_new_r48_calibrated_temp.jsonl \
  --csv outputs/flood_stage2_prep_r48/stage2_samples.csv \
  --out_dir outputs/vis_predictions_overlay \
  --max_outputs 200 \
  --fill_opacity 0.5
```

```bash
python3 scripts/visualize_stage2_overlays.py \
  --pred_jsonl outputs/inference_ensemble_top3_new_r48_calibrated_temp.jsonl \
  --csv outputs/flood_stage2_prep_r48/stage2_samples.csv \
  --out_dir outputs/vis_predictions_overlay_gt2 \
  --gt_labels 2 \
  --max_outputs 100 \
  --fill_opacity 0.5
```

```bash
python3 scripts/visualize_stage2_overlays.py \
  --pred_jsonl outputs/inference_ensemble_top3_new_r48_calibrated_temp.jsonl \
  --csv outputs/flood_stage2_prep_r48/stage2_samples.csv \
  --out_dir outputs/vis_predictions_overlay_gt3 \
  --gt_labels 3 \
  --max_outputs 100 \
  --fill_opacity 0.5
```

Status:
- Visual overlay kit is functional and produces interpretable per-sample diagnostics for uncertainty-aware review.

