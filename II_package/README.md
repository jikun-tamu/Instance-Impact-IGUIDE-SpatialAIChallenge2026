## Instance Impact Package (`II_package`)

Portable runtime bundle for the full instance-impact workflow:

- Stage 1: building instance extraction (SAM3)
- Shared subimage generation (`256x256`, `mask_M`, `mask_R`)
- Stage 2a: population/type inference
- Stage 2b: damage ensemble inference (with calibration)
- Presentation + instance-level overlays

This package intentionally **does not duplicate large imagery/output assets**.  
You provide pre/post images at runtime.

### Package Layout

- `scripts/` - runtime scripts (driver + stage scripts)
- `stage1/SAM3_Final_20260226/` - Stage 1 code used by the driver
- `models/stage2a/` - Stage 2a checkpoint
- `models/stage2b/` - 3 Stage 2b inference checkpoints
- `configs/stage2b/` - matching train configs for Stage 2b models
- `calibration/` - per-checkpoint calibration artifacts
- `docs/` - project docs (`README`, `Stage2b`, `stage1`, `stage2a`)

### Recommended Full Run (Baseline)

From inside `II_package/` (recommended baseline we have been using):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON_BIN="$(which python3)" \
bash run_pipeline.sh \
  --pre_image /path/to/<tile>_pre_disaster.png \
  --post_image /path/to/<tile>_post_disaster.png \
  --run_id e2e_demo \
  --stage1_output_style notebook \
  --stage1_tile_size 512 \
  --stage1_overlap 64 \
  --stage1_min_size 30 \
  --stage1_device cuda:0 \
  --stage2a_device cuda \
  --stage2b_device cuda
```

### Packaged Example Pair Command

Use the bundled sample pair in `example_image_pair/`:

```bash
cd /anvil/scratch/x-jliu7/ProjII/II_package

CUDA_VISIBLE_DEVICES=0 PYTHON_BIN="$(which python3)" \
bash run_pipeline.sh \
  --pre_image /anvil/scratch/x-jliu7/ProjII/II_package/example_image_pair/nepal-flooding_00000408_pre_disaster.png \
  --post_image /anvil/scratch/x-jliu7/ProjII/II_package/example_image_pair/nepal-flooding_00000408_post_disaster.png \
  --run_id e2e_example_pair_00000408 \
  --stage1_output_style notebook \
  --stage1_tile_size 512 \
  --stage1_overlap 64 \
  --stage1_min_size 30 \
  --stage1_device cuda:0 \
  --stage2a_device cuda \
  --stage2b_device cuda
```

### Standalone Presentation Notebook

Notebook path:

- `present_instance_results_notebook.ipynb`

Purpose:

- Run end-to-end inference on a custom pre/post pair
- Read a completed run under `outputs/driver_runs/<run_id>/`
- Print summary statistics
- Show uncertainty table previews
- Render a few overlay images

Server-kernel usage (e.g., `geoai`):

- Open the notebook and select the `geoai` kernel.
- In the config cell, set:
  - `PACKAGE_ROOT` (usually auto-detected)
  - `RUN_ID` (the run folder to present)
- Configure input paths and runtime options in the notebook config cell.
- Run cells top-to-bottom.

This notebook is designed as a full pipeline narrative from input images to final outputs.

The launcher calls `scripts/run_instance_impact_driver.py` with package-local defaults:

- Stage 2a ckpt: `models/stage2a/stage2a_best_model.pt`
- Stage 2b ckpts: `models/stage2b/inference0.7273.pt`, `inference0.7066_seed9999.pt`, `inference0.7034_seed7777.pt`
- Stage 2b calibration: `temperature` using bundled calibration dirs
- Shared crop protocol: `crop_size=256`, `ring_radius_px=48`
- Stage 2b ensemble weights: `4,3,2`

### Full-Run Flag Guide

Use `bash run_pipeline.sh --help` for the full list. The key flags are:

- `--pre_image`, `--post_image` (required): input pair for one tile.
- `--run_id`: output folder suffix under `outputs/driver_runs/`.
- `--out_root`: output root (default `outputs/driver_runs`).
- `--overwrite_run_dir`: replace an existing run folder with same `run_id`.

Stage 1:

- `--stage1_output_style`: keep as `notebook` (recommended; produces expected instance JSON path).
- `--stage1_tile_size`: keep `512` for stable detection on flood scenes.
- `--stage1_overlap`: keep `64` to reduce boundary misses.
- `--stage1_min_size`: keep `30` (lower values add noise; higher values can miss small buildings).
- `--stage1_device`: usually `cuda:0`.
- `--stage1_prompt`: default `building` (recommended).

Shared subimage generation:

- `--crop_size`: keep `256` (Stage 2a/2b were wired around this).
- `--ring_radius_px`: keep `48` (frozen Stage 2b baseline).
- `--shared_num_workers`: increase only if you have spare CPU.

Stage 2a:

- `--stage2a_ckpt`: default packaged checkpoint is recommended.
- `--stage2a_batch_size`: default `64` is generally safe.
- `--stage2a_device`: use `cuda` when available.

Stage 2b:

- `--stage2b_ckpts`: keep the packaged top-3 checkpoints unless intentionally swapping ensemble members.
- `--stage2b_configs`: keep aligned with `--stage2b_ckpts` (same order).
- `--stage2b_weights`: keep `4,3,2` (frozen default).
- `--stage2b_calibration_method`: keep `temperature` (recommended).
- `--stage2b_calibration_dirs`: keep bundled dirs aligned with checkpoint order.
- `--stage2b_batch_size`: default `64` (reduce only if GPU memory is limited).

Runtime logging:

- `--verbose`: print full logs from all stages.
- Default (without `--verbose`): quiet mode, prints only presenter `[summary]` lines and final `[done]` output paths.

Presentation / visualization:

- `--top_k_uncertain`: number of uncertain rows written to CSV.
- `--print_top_n`: number of uncertain rows printed in terminal.
- `--visualize_max_outputs`: cap number of overlay images.
- `--visualize_fill_opacity`: overlay fill alpha (default `0.5`).
- `--visualize_gt_labels`: optional label filter for overlays (for focused QA).

### Defaults You Should Usually Not Change

For reproducible baseline behavior, keep these fixed unless you are intentionally experimenting:

- `--stage1_output_style notebook`
- `--stage1_tile_size 512`
- `--stage1_overlap 64`
- `--stage1_min_size 30`
- `--crop_size 256`
- `--ring_radius_px 48`
- `--stage2b_weights 4,3,2`
- `--stage2b_calibration_method temperature`
- packaged Stage 2a and Stage 2b checkpoints/config/calibration mappings

### Main Outputs

For `--run_id <id>`, outputs go to:

- `outputs/driver_runs/<id>/stage1/`
- `outputs/driver_runs/<id>/shared_instances_r48/shared_instance_samples.csv`
- `outputs/driver_runs/<id>/stage2a_predictions.csv`
- `outputs/driver_runs/<id>/stage2b_ensemble.jsonl`
- `outputs/driver_runs/<id>/instance_results_presented.csv`
- `outputs/driver_runs/<id>/instance_results_top_uncertain.csv`
- `outputs/driver_runs/<id>/vis_instance_level/`

### Dependencies and Environment Export

Core runtime dependencies used by this package:

- Python `3.10`
- `torch`, `torchvision`, `torchaudio`, `timm`
- `numpy`, `scipy`, `pillow`
- `shapely`, `rasterio`
- `segment-geospatial`, `geoai-py`, `huggingface_hub`

### Using Exported Env Files (`.yml` / `.txt`)

If you already have exported environment files in `II_package/`, recreate with:

```bash
conda env create -f II_package/environment.yml
# or:
conda create -n ii_package python=3.10
pip install -r II_package/requirements.txt
```

### Notes

- Stage 1 requires SAM3 dependencies and model access (HF token if needed).
- All defaults are overrideable via driver flags, but baseline runs should follow the recommended command above.
- This folder is self-contained for code + models + calibration; only input images are external.

### Quick Troubleshooting

- **HF / gated model access error (Stage 1)**
  - Symptom: Hugging Face 401/403 or gated repo error for SAM3.
  - Fix:
    ```bash
    export HF_TOKEN=...
    ```
  - Ensure your account has access to the required SAM3 model.

- **CUDA not available / GPU init issues**
  - Symptom: falls back to CPU or reports CUDA init/runtime errors.
  - Fix: pin one visible GPU and set devices explicitly:
    ```bash
    CUDA_VISIBLE_DEVICES=0 PYTHON_BIN="$(which python3)" \
    bash run_pipeline.sh ... --stage1_device cuda:0 --stage2a_device cuda --stage2b_device cuda
    ```
  - If GPU remains unstable, rerun in a fresh session.

- **Out-of-memory (OOM) during Stage 2a/2b inference**
  - Symptom: CUDA OOM error in Stage 2a or Stage 2b.
  - Fix: reduce batch sizes first (safe fallback):
    ```bash
    bash run_pipeline.sh ... --stage2a_batch_size 16 --stage2b_batch_size 16
    ```
  - If still OOM, try `8`.

- **Stage 1 finds too few buildings**
  - Symptom: very small instance count on a tile with visible buildings.
  - Fix: keep recommended tiling/min-size defaults:
    - `--stage1_output_style notebook`
    - `--stage1_tile_size 512`
    - `--stage1_overlap 64`
    - `--stage1_min_size 30`

- **Run folder already exists**
  - Symptom: rerun collisions or mixed old/new outputs.
  - Fix: either choose a new `--run_id` or pass:
    ```bash
    --overwrite_run_dir
    ```
  - To fully clean generated artifacts:
    ```bash
    rm -rf outputs
    ```
