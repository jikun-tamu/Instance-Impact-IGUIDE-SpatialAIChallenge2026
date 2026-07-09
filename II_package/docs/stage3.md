# Stage-3 Impact Synthesis Record

## Executive Summary

- Stage status: Stage-3 is implemented as a driver, presentation, and aggregation layer. It is not a separately trained impact model.
- Core task: align Stage-1 building instances, Stage-2a exposure predictions, and Stage-2b damage predictions into instance-level impact products.
- Current default single-pair path: `run_instance_impact_driver.py` orchestrates Stage 1 -> shared crops -> Stage 2a -> Stage 2b -> presenter -> overlays.
- Current default presentation path: `present_instance_results.py` merges by building instance uid and writes a reviewable CSV plus uncertainty-focused rows.
- Current default multi-date path: `aggregate_multidate_predictions.py` aggregates Stage-2b predictions across multiple post-disaster dates.
- Current operational rule for real-world multi-date assessment: M2b, a coverage-aware majority vote with not-identifiable handling.
- Main caveat: `driver_exposure_damage_score = pred_population * expected_severity` is a heuristic ranking signal, not a validated causal or trained impact metric.

## Objective

Stage-3 converts model outputs into a building-instance impact representation that can be inspected, ranked, visualized, and summarized.

The Stage-3 task assumes:

- input: a stable building instance id from Stage 1/shared artifacts;
- exposure signal: Stage-2a population and type predictions;
- damage signal: Stage-2b ordinal damage class, expected severity, calibrated confidence, and ensemble uncertainty;
- output: one row per building instance with joined upstream signals and optional qualitative overlays.

This layer is deliberately CLI-first and artifact-first. It should preserve enough intermediate columns for debugging joins, quality control, and manuscript/result tracing.

## Current Scope

- Single pre/post image-pair synthesis through `scripts/driver/run_instance_impact_driver.py`.
- Standalone presentation merge through `scripts/stage3/present_instance_results.py`.
- Instance-level overlay rendering through `scripts/stage3/visualize_stage2_overlays.py`.
- Multi-date Stage-2b aggregation through `scripts/stage3/aggregate_multidate_predictions.py`.
- LA Fire 2025 recorded as a zero-shot/domain-mismatch application example, not as validated wildfire performance.

## Current Scripts

- `scripts/driver/run_instance_impact_driver.py`
- `scripts/stage3/present_instance_results.py`
- `scripts/stage3/visualize_stage2_overlays.py`
- `scripts/stage3/aggregate_multidate_predictions.py`
- Supporting adapter: `scripts/stage2a/infer/build_stage2a_infer_csv.py`

## End-to-End Pipeline Details

### Single-Pair Driver Path

The single-pair driver is:

```text
pre/post image pair
  -> Stage-1 SAM3 building inference
  -> generate_shared_instance_subimages.py
  -> build_stage2a_infer_csv.py
  -> infer_stage2a.py
  -> infer_stage2_ensemble.py
  -> present_instance_results.py
  -> visualize_stage2_overlays.py
```

The driver writes one run directory under `outputs/driver_runs/<run_id>` by default. Important artifacts are:

- `stage1/`: Stage-1 masks, confidence rasters, and prediction JSON.
- `shared_instances_r48/shared_instance_samples.csv`: shared instance crop/mask table.
- `stage2a_infer_input.csv`: Stage-2a adapter CSV.
- `stage2a_predictions.csv`: Stage-2a exposure/type predictions.
- `stage2b_ensemble.jsonl`: Stage-2b ensemble damage predictions.
- `instance_results_presented.csv`: Stage-3 merged presentation table.
- `instance_results_top_uncertain.csv`: high-uncertainty Stage-2b cases for review.
- `vis_instance_level/`: per-instance overlay PNGs.

The driver default Stage-2b package paths are:

```text
models/stage2b/inference0.7273.pt
models/stage2b/inference0.7066_seed9999.pt
models/stage2b/inference0.7034_seed7777.pt
configs/stage2b/run019_seed2025_train_config.json
configs/stage2b/seed9999_train_config.json
configs/stage2b/seed7777_train_config.json
calibration/calibration_run019_r48
calibration/calibration_seed9999_r48
calibration/calibration_seed7777_r48
```

The default ensemble weights are `4,3,2`, and the default calibration method is `temperature`.

### Shared Instance Contract

`shared_instance_samples.csv` is the Stage-1-to-Stage-2 bridge. Stage-3 uses it as the source of instance identity, crop paths, mask paths, and Stage-1 confidence.

Important columns:

- `bldg_uid`: canonical building instance id for Stage 1 and Stage 2b joins.
- `tile_id`, `event_id`, `hazard_type`: metadata.
- `sam3_confidence`: Stage-1 segmentation confidence when present.
- `pre_crop`, `post_crop`: per-building RGB crop paths.
- `mask_M`: target building mask path.
- `mask_R`: surrounding ring mask path.
- `m_area_px`, `r_area_px`: footprint and ring pixel areas.
- `crop_x0`, `crop_y0`, `crop_size`, `cx`, `cy`: crop geometry/debug fields.

`build_stage2a_infer_csv.py` adapts this shared table to the Stage-2a input schema. It maps:

- `bldg_uid` -> `building_uid`
- `pre_crop` -> `crop_path`
- `mask_M` -> `mask_path`

It also carries through metadata and derives deploy-visible geometry columns such as `mask_area_px`, `mask_fill_ratio`, `bbox_aspect_ratio`, `geometry_compactness`, `footprint_m2`, and `log_footprint_m2` when requested.

### Stage-2a Input To Stage-3

Stage-3 expects Stage-2a predictions in CSV form from `infer_stage2a.py`.

Join key:

- Stage-2a: `building_uid`
- Stage-1/shared and Stage-2b: `bldg_uid`

Important Stage-2a columns consumed by the presenter:

- `pred_population`
- `pred_log1p_population`
- `pred_type_idx`
- `pred_type_class`
- `pred_type_conf`
- `tile_id`
- `event_id`
- `sam3_confidence`
- `hazard_type`

Stage-3 does not reinterpret the Stage-2a taxonomy. It reports whatever the Stage-2a checkpoint and inference CSV produce. For the current Stage-2a drop-both direction, type predictions are closed-world over `residential_small`, `residential_multi`, and `commercial`; dropped `institutional` and `other` rows are a known coverage limitation of that upstream policy.

### Stage-2b Input To Stage-3

Stage-3 expects Stage-2b damage predictions in JSONL form from `infer_stage2_ensemble.py`.

Join key:

- `bldg_uid`

Important Stage-2b fields consumed or preserved:

- `y_pred_ensemble`
- `expected_severity_ensemble`
- `pmax`
- `margin`
- `entropy`
- `var_predicted_class_prob_weighted`
- `var_expected_severity_weighted`
- `calibration_method`
- `ensemble_probs`
- `ensemble_probs_calibrated`

The damage label space is the Stage-2b ordinal space:

| class | semantic role |
|---:|---|
| 0 | no visible damage |
| 1 | minor damage |
| 2 | major damage |
| 3 | destroyed / highest severity |

`expected_severity_ensemble` is computed from calibrated ensemble probabilities as the expected ordinal class value.

### Presentation Merge

`present_instance_results.py` builds a union of instance ids found in:

- shared Stage-1 rows keyed by `bldg_uid`;
- Stage-2a rows keyed by `building_uid`;
- Stage-2b rows keyed by `bldg_uid`.

It writes one row per id and records join flags:

- `has_stage1`
- `has_stage2a`
- `has_stage2b`

The core presentation fields are:

| Field | Meaning |
|---|---|
| `instance_id` | Canonical building instance uid |
| `tile_id`, `event_id`, `hazard_type` | Metadata propagated from upstream rows |
| `stage1_sam3_confidence` | Stage-1 confidence |
| `stage2a_pred_population` | Stage-2a exposure proxy |
| `stage2a_pred_log1p_population` | Log-space exposure prediction |
| `stage2a_pred_type_idx` | Stage-2a type index |
| `stage2a_pred_type_class` | Stage-2a type name |
| `stage2a_pred_type_conf` | Stage-2a max type probability |
| `stage2b_pred_damage_class` | Stage-2b ensemble damage class |
| `stage2b_expected_severity` | Stage-2b calibrated expected ordinal severity |
| `stage2b_pmax` | Stage-2b max calibrated probability |
| `stage2b_margin` | Difference between top two calibrated probabilities |
| `stage2b_entropy` | Entropy of calibrated probabilities |
| `stage2b_var_predicted_class_prob_weighted` | Ensemble disagreement on predicted class probability |
| `stage2b_var_expected_severity_weighted` | Ensemble disagreement on expected severity |
| `stage2b_calibration_method` | Calibration method used by Stage-2b inference |
| `driver_exposure_damage_score` | Heuristic `population * expected severity` ranking score |
| `pre_crop`, `post_crop`, `mask_M`, `mask_R` | Debug and visualization pointers |

The presenter also prints and optionally writes summary statistics:

- join coverage;
- counts by Stage-2a type;
- counts by Stage-2b damage class;
- quantiles for Stage-1 confidence;
- quantiles for Stage-2a population/type confidence;
- quantiles for Stage-2b pmax, margin, entropy, and severity variance;
- quantiles for `driver_exposure_damage_score`.

### Heuristic Impact Ranking

The current driver score is:

```text
driver_exposure_damage_score = stage2a_pred_population * stage2b_expected_severity
```

Interpretation:

- higher score means a building has larger predicted exposure and/or higher predicted damage severity;
- it is useful for triage, ranking, and qualitative review;
- it is not trained, calibrated, or validated as a final human impact metric.

Because Stage-2a population is a proxy label and Stage-2b expected severity is an ordinal model output, this product must be described as a heuristic exposure-damage ranking score in reports or papers.

### Top-Uncertain Review Table

`instance_results_top_uncertain.csv` is sorted by:

1. high `stage2b_entropy`,
2. low `stage2b_pmax`,
3. high `stage2b_var_expected_severity_weighted`.

This table is intended for QA, human inspection, and model debugging. It should not be interpreted as a separate prediction product.

## Commands

### End-To-End Single-Pair Driver

Use this pattern from `II_package/`:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/driver/run_instance_impact_driver.py \
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
  --stage2a_infer_mode native3_ensemble \
  --stage2b_calibration_method temperature
```

The driver has `--dry_run` for printing the exact subcommands without executing them, and `--verbose` for full subprocess logs.

### Standalone Presentation Merge

```bash
python3 scripts/stage3/present_instance_results.py \
  --shared_csv outputs/driver_runs/<run_id>/shared_instances_r48/shared_instance_samples.csv \
  --stage2a_csv outputs/driver_runs/<run_id>/stage2a_predictions.csv \
  --stage2b_jsonl outputs/driver_runs/<run_id>/stage2b_ensemble.jsonl \
  --out_csv outputs/driver_runs/<run_id>/instance_results_presented.csv \
  --out_summary_json outputs/driver_runs/<run_id>/instance_results_summary.json \
  --out_top_uncertain_csv outputs/driver_runs/<run_id>/instance_results_top_uncertain.csv \
  --top_k_uncertain 30 \
  --print_top_n 15
```

### Instance-Level Visualization

```bash
python3 scripts/stage3/visualize_stage2_overlays.py \
  --pred_jsonl outputs/driver_runs/<run_id>/stage2b_ensemble.jsonl \
  --csv outputs/driver_runs/<run_id>/shared_instances_r48/shared_instance_samples.csv \
  --stage2a_csv outputs/driver_runs/<run_id>/stage2a_predictions.csv \
  --out_dir outputs/driver_runs/<run_id>/vis_instance_level \
  --max_outputs 100 \
  --fill_opacity 0.5
```

Overlay layout:

- top-left panel: Stage-2b predicted class, ground truth if present, expected severity, pmax, margin, entropy;
- bottom-left panel: Stage-1 confidence;
- bottom-right panel: Stage-2a population, type, and type confidence.

The mask fill color follows Stage-2b predicted damage class:

- class `0`: green;
- class `1`: yellow-green;
- class `2`: orange;
- class `3`: red.

## Multi-Date Aggregation

### Input Directory Contract

`aggregate_multidate_predictions.py` expects:

```text
<cell_run_dir>/
  dates/
    <YYYYMMDD>/
      stage2b_<YYYYMMDD>.jsonl
      quality_metrics.json
      shared_for_date.csv
```

For each date:

- `stage2b_*.jsonl` supplies per-building Stage-2b predictions.
- `quality_metrics.json` supplies `tile_quality_ok`.
- `shared_for_date.csv` supplies per-building `quality_ok` for crop coverage.

### Aggregation Outputs

The script writes:

- `aggregated_predictions.jsonl`
- `aggregated_predictions.csv`

Each output row is keyed by `bldg_uid` and records:

- date counts and date lists;
- tile-rejected dates;
- coverage-invalid dates;
- M1/M1b/M2/M2b/M3 aggregate classes;
- probability vectors for probability-average methods;
- M2b vote counts;
- label entropy;
- `is_unstable`;
- per-date summaries.

### Aggregation Methods

| Method | Rule | Not-identifiable behavior |
|---|---|---|
| M1 | Average calibrated probabilities across tile-quality-ok dates | `-1` if no tile passes |
| M1b | Average calibrated probabilities across dates with tile quality and crop coverage | `-1` if no valid coverage |
| M2 | Majority vote across tile-quality-ok dates | `-1` if no tile passes |
| M2b | Majority vote across dates with tile quality and crop coverage | `-1` if no valid coverage |
| M3 | Probability average requiring tile and crop quality, with fallback to M1 | falls back to M1 |

M2b is the current real-world rule:

1. A date is valid for a building only when `tile_quality_ok=True` and `crop_quality_ok=True`.
2. If zero dates are valid, the building is `-1` / not identifiable.
3. If one or more dates are valid, aggregate valid-date damage labels by majority vote.
4. If the vote ties, use the highest tied damage class as a conservative tie-break.

### Multi-Date Command

```bash
python3 scripts/stage3/aggregate_multidate_predictions.py \
  --cell_run_dir la_fire_results/demo_cell_00507/multidate_inputs \
  --out_jsonl la_fire_results/demo_cell_00507/aggregated_predictions.jsonl \
  --out_csv la_fire_results/demo_cell_00507/aggregated_predictions.csv
```

## Recorded Example Results

### Flood-Domain Driver Checks

The framework README records completed end-to-end runs on:

- `nepal-flooding_00000442`: destroyed-heavy flood tile;
- `nepal-flooding_00000408`: flood tile with minor and major labels.

Generated outputs include `stage1/`, shared crops/masks, `stage2a_predictions.csv`, `stage2b_ensemble.jsonl`, `instance_results_presented.csv`, `instance_results_top_uncertain.csv`, and `vis_instance_level/`.

### LA Fire 2025 Multi-Date Example

Recorded result file:

```text
la_fire_results/damage_summary.md
```

Recorded M2b summary:

| Metric | Value |
|---|---:|
| Cells processed with buildings | 120 of 295 |
| Cells with zero Stage-1 detections | 175 |
| Total building instances | 21,797 |
| Post-disaster dates evaluated | 5 |

M2b damage distribution:

| Damage class | Count | Percent |
|---|---:|---:|
| No damage (0) | 16,246 | 74.5% |
| Minor (1) | 4,174 | 19.1% |
| Major (2) | 39 | 0.2% |
| Destroyed (3) | 93 | 0.4% |
| Unknown / no valid imagery (-1) | 1,245 | 5.7% |

Important caveat: this is a no-retraining wildfire application using a flood-trained Stage-2b model. It is a demonstration of the instance-impact workflow and multi-date aggregation mechanics, not validated wildfire damage performance.

## QA And Review Checks

Recommended checks before using a Stage-3 run in results:

- Confirm `instance_results_presented.csv` has the expected number of rows.
- Confirm `join_coverage.all_three` is close to the intended run population.
- Inspect rows where `has_stage1`, `has_stage2a`, or `has_stage2b` is zero.
- Review `instance_results_top_uncertain.csv` and several overlay PNGs.
- Check Stage-2b damage class distribution for obvious domain or nodata artifacts.
- For multi-date runs, prefer M2b over M1 when crop coverage can fail.
- Verify that `-1` not-identifiable buildings are preserved and not silently mapped to damage class `0`.
- Treat `driver_exposure_damage_score` as ranking only.

## Known Limitations

- Stage-3 has no learned impact model; it is a synthesis and reporting layer.
- `driver_exposure_damage_score` is heuristic and combines two upstream model/proxy outputs.
- Stage-2a population is a proxy-label prediction, not observed building occupancy.
- Current Stage-2a drop-both type direction has a closed-world three-class limitation.
- Stage-2b checkpoints are flood-domain; wildfire or other hazard applications are domain-mismatch unless retrained or validated.
- Multi-date aggregation assumes stable `bldg_uid` identity across dates.
- Per-date crop coverage quality must be provided for M1b/M2b to behave as intended.
- `generate_shared_instance_subimages.py` currently skips some `MULTIPOLYGON` WKT rows, which can reduce instance count.
- Model weights are local/package artifacts and may be ignored by git; submission-ready packaging needs a manifest, checksums, or download mechanism.

## Checklist Status

- [x] Single-pair driver assembled.
- [x] Stage-1/shared/Stage-2a/Stage-2b presentation merge implemented.
- [x] Instance-level overlay visualization implemented.
- [x] Stage-2b multi-date aggregation implemented.
- [x] M2b coverage-aware real-world rule documented in code and summary files.
- [ ] Add focused tests for `present_instance_results.py`.
- [ ] Add focused tests for M1/M1b/M2/M2b/M3 edge cases.
- [ ] Add submission-ready model artifact manifest and checksums.
- [ ] Decide manuscript wording for heuristic exposure-damage score.
