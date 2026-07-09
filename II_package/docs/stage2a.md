# Stage-2a Three-Class Exposure Pipeline Record

## 0. Executive Summary

Stage-2a currently uses a **closed-world drop-both three-class exposure pipeline** for downstream Stage-2b / Stage-3 integration.

The selected production-facing path is:

```text
Drop institutional + other
        ↓
Native three-class type model
residential_small / residential_multi / commercial
        ↓
Population regression
log_footprint_m2 + soft predicted 3-class type probabilities + residual HGB
        ↓
pred_type_class, pred_type_probs, pred_population
```

### Current final contract

| Component | Selected contract |
|---|---|
| Label policy | `drop_institutional_other` |
| Active classes | `residential_small`, `residential_multi`, `commercial` |
| Retained rows | `21,428 / 23,014 = 0.9311` |
| Type model | ConvNeXt-Tiny, native 3-logit head, RGB + mask + deployable geometry, `mask_m` pooling |
| Population model | `log_footprint_m2 + soft native predicted type + residual HGB on ridge geometry` |
| Primary fixed-split result | M32 native five-seed population stability |
| Same-split ensemble check | M33 native five-seed probability ensemble |
| Split-sensitivity check | M41 native alternative tile-blocked splits |
| Deployment route | `infer_stage2a_native3_ensemble.py` + packaged residual-HGB population artifact |

### Main results

| Result block | Purpose | Key result | Interpretation |
|---|---|---:|---|
| M20 | Taxonomy selection | Drop-both 3-class macro-F1 `0.8628` | Three-class reduced taxonomy is substantially cleaner than four-class policies. |
| M31 | Native type seed stability | Macro-F1 `0.8529 ± 0.0106` | Final native type model is stable across seeds on fixed `m20_split`. |
| M32 | Native population seed stability | Log MAE `0.0760 ± 0.0032`, factor-2 `0.9603 ± 0.0034` | Primary fixed-split native population claim. |
| M33 | Native same-split ensemble | Type ECE `0.0342`, population log MAE `0.0734` | Best native same-split ensemble check; not a multi-run stability claim. |
| M34 | Native type ablation | RGB+mask `0.8626`, RGB-only `0.8048` | Mask input and imbalance handling matter; geometry is not the only signal. |
| M35 | Native population type contribution | With type `0.0742`, no-type residual `0.1157` | Predicted type gives large deployable gain beyond footprint alone. |
| M41 | Split sensitivity + calibration | alt1 log MAE `0.1041`, alt2 `0.0613`; calibration worsens population | Cross-split variation is larger than same-split seed variation; raw ensemble probabilities remain default. |

### Claim boundary

Supported claims:

- Stage-2a has a documented **three-class closed-world** type-to-population pipeline.
- M31/M32 show **fixed `m20_split` seed stability**, not broad cross-split robustness.
- M33 shows a useful **same-split ensemble improvement** and is the packaged inference default.
- M41 shows that the pipeline remains usable under alternative tile-blocked splits, but split composition materially changes metrics.
- Population metrics are **proxy-label fidelity metrics** for `estimated_population`.

Unsupported claims:

- This is not an independent held-out test result.
- This is not an all-building-type model; `institutional` and `other` are out of scope for the selected contract.
- This is not independent validation of observed per-building occupancy.
- M33 should not be described as a multi-run stability result; it is one same-split ensemble result.

---

## 1. Dataset, Split, and Target Definition

### 1.1 Label policy and split

| Field | Value |
|---|---|
| Base label file | `outputs/stage2a/ml_dataset/labels_manifest.csv` |
| Drop-both manifest | `outputs/stage2a/ml_dataset/labels_manifest_m20_drop_institutional_other.csv` |
| Policy | `drop_institutional_other` |
| Dropped classes | `institutional`, `other` |
| Active classes | `residential_small`, `residential_multi`, `commercial` |
| Rows before / after | `23,014 -> 21,428` |
| Retained fraction | `0.9311` |
| Main split | fixed `m20_split` |
| Split group | tile-blocked by `tile_base` |
| Validation ratio | `0.15` |
| Test ratio | `0.0` |

Main manifest construction command:

```bash
PYTHONPATH=. python scripts/stage2a/training/prepare_stage2a_label_policy_manifest.py \
  --labels_csv outputs/stage2a/ml_dataset/labels_manifest.csv \
  --out_manifest outputs/stage2a/ml_dataset/labels_manifest_m20_drop_institutional_other.csv \
  --policy drop_institutional_other \
  --split_col m20_split \
  --split_group_col tile_base \
  --split_seed 2025 \
  --val_ratio 0.15 \
  --test_ratio 0.0
```

### 1.2 Population target provenance

The Stage-2a population label is `estimated_population`, not direct observed occupancy.

The upstream Harris CBG workflow constructs this target as an exposure proxy:

```text
estimated_population = estimated_units * people_per_unit_ratio * occupancy_rate
```

Important implications:

- For multi-family buildings, `estimated_units` is partly footprint-derived.
- `footprint_m2` / `log_footprint_m2` shares information with the target construction.
- This feature is valid and deployable, but it should be described as predicting a footprint/type-derived exposure proxy.
- Population metrics should be interpreted as **proxy-label fidelity**, not independent validation against observed building population.

---

## 2. Final Native Three-Class Pipeline

This section states the final selected model contract before the experiment history.

### 2.1 Native type model contract

```text
backbone = ConvNeXt-Tiny
head = native 3-logit classifier
classes = residential_small, residential_multi, commercial
input = RGB crop + building mask + deployable geometry
pooling = mask_m
loss = cross entropy
sampler = weighted
class weights = inverse-sqrt
```

No final native model emits `prob_institutional` or `prob_other`.

### 2.2 Population model contract

The selected population model is a tabular regression model:

```text
features = log_footprint_m2 + soft native predicted 3-class type probabilities
model = residual HGB on ridge geometry
output = pred_population
```

This is a regression model for continuous `estimated_population`.

### 2.3 Packaged inference route

The packaged route uses the five M31 native strict type checkpoints, averages their three-class probabilities, and applies the packaged residual-HGB population model.

Packaged artifact paths:

```text
models/stage2a/native3_drop_both_ensemble/seed42/stage2a_best_model.pt
models/stage2a/native3_drop_both_ensemble/seed123/stage2a_best_model.pt
models/stage2a/native3_drop_both_ensemble/seed2025/stage2a_best_model.pt
models/stage2a/native3_drop_both_ensemble/seed3407/stage2a_best_model.pt
models/stage2a/native3_drop_both_ensemble/seed7777/stage2a_best_model.pt
models/stage2a/native3_drop_both_ensemble/population_logfootprint_soft_type_residual_hgb.pkl
models/stage2a/native3_drop_both_ensemble/population_logfootprint_soft_type_residual_hgb_summary.json
```

Packaging validation check:

| Metric | Value |
|---|---:|
| Population log MAE | `0.073351` |
| Factor-2 hit rate | `0.960674` |

Default inference command pattern:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/stage2a/infer/infer_stage2a_native3_ensemble.py \
  --input_csv outputs/shared_instances_sanity_2tiles_v2_r48/stage2a_infer_input.csv \
  --ckpts models/stage2a/native3_drop_both_ensemble/seed42/stage2a_best_model.pt,models/stage2a/native3_drop_both_ensemble/seed123/stage2a_best_model.pt,models/stage2a/native3_drop_both_ensemble/seed2025/stage2a_best_model.pt,models/stage2a/native3_drop_both_ensemble/seed3407/stage2a_best_model.pt,models/stage2a/native3_drop_both_ensemble/seed7777/stage2a_best_model.pt \
  --population_model_pkl models/stage2a/native3_drop_both_ensemble/population_logfootprint_soft_type_residual_hgb.pkl \
  --out_csv outputs/shared_instances_sanity_2tiles_v2_r48/stage2a_native3_ensemble_predictions.csv \
  --batch_size 64 \
  --num_workers 4 \
  --device cuda
```

M41 showed that post-hoc temperature scaling improves type-side ECE/NLL but worsens downstream population log MAE. Therefore temperature scaling remains diagnostic/optional only; the default population inference path uses raw M33 ensemble probabilities.

---

## 3. Main Three-Class Experiment Narrative

The main experimental story is ordered by decision logic rather than raw completion time:

```text
M20   choose reduced three-class taxonomy
M23   compare population model families after drop-both selection
M24   ablate population features and model form
M30   isolate predicted-type contribution from residual-HGB architecture
M25   historical compatibility-head type stability
M26   historical compatibility-head population stability
M28   historical compatibility-head ensemble check
M31-M35 replace compatibility head with final native three-class head
M41   final native split sensitivity, calibration, and uncertainty diagnostics
```

---

### 3.1 M20: Three-Class Taxonomy Selection

**Question.** Does the drop-both three-class taxonomy produce a cleaner type task than broader policies?

Main selected result:

| Policy | Active classes | Rows kept | Best recipe | Val macro-F1 | Val accuracy |
|---|---:|---:|---|---:|---:|
| Drop both | 3 | `21,428` | ConvNeXt weighted CE + `mask_m` | `0.8628` | `0.9304` |

Interpretation:

- Dropping both `institutional` and `other` produces the cleanest type task.
- The selected pipeline is intentionally closed-world over three classes.
- Detailed non-selected label-policy results are moved to Appendix A.

---

### 3.2 M23: Population Model Family Selection

**Question.** After selecting the three-class path, do predicted type probabilities improve population prediction, and do direct image-pixel population models help?

Protocol summary:

- Type source: M20 drop-both ConvNeXt-Tiny weighted CE `mask_m` type model.
- Train-row type features: tile-blocked OOF type predictions.
- Validation type features: full-train validation predictions.
- Population candidates: HGB, ExtraTrees, Ridge, residual HGB on ridge geometry.
- Pixel baselines: population-only EfficientNet-B0 and ConvNeXt-Tiny.

Top results:

| Family | Model | Features | Val log MAE | Factor-2 | MAE | R2 |
|---|---|---|---:|---:|---:|---:|
| Tabular | residual HGB on ridge geometry | core geometry + soft predicted type | `0.0812` | `0.9607` | `16.5` | `0.8281` |
| Tabular | HGB | core geometry + hard predicted type | `0.0841` | `0.9566` | `22.3` | `0.764` |
| Pixel | ConvNeXt pop-only | RGB+mask+geometry | `0.2889` | `0.9079` | `72.4` | `0.482` |
| Pixel | EfficientNet-B0 pop-only | RGB+mask+geometry | `0.4201` | `0.8246` | `102.6` | `0.466` |
| Pixel | EfficientNet-B0 pop-only | RGB+mask+geometry, `mask_m` | `0.4379` | `0.8037` | `108.8` | `0.293` |

Interpretation:

- Tabular geometry + predicted type is much stronger than direct image-to-population regression.
- Population prediction is best framed as a deployable geometry/type regression problem, not a raw pixel regression problem.

---

### 3.3 M24: Population Feature and Model Ablation

**Question.** Which components of the M23 population model matter?

Top ablation findings:

| Rank | Group | Variant | Model | Val log MAE | Delta vs M23 selected | MAE | R2 |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | geometry | `log_footprint_only` | residual HGB + soft type | `0.0738` | `-0.0073` | `14.6` | `0.846` |
| 2 | geometry | `core_drop_bbox_aspect_ratio` | residual HGB + soft type | `0.0781` | `-0.0030` | `16.1` | `0.825` |
| 3 | geometry | `core_drop_mask_fill_ratio` | residual HGB + soft type | `0.0796` | `-0.0016` | `16.0` | `0.830` |
| 5 | selected | `core_soft_all_residual_hgb` | residual HGB + soft type | `0.0812` | `0.0000` | `16.5` | `0.828` |
| 6 | type | `hard_one_hot` | residual HGB + hard type | `0.0812` | `+0.0000` | `16.4` | `0.817` |
| 10 | model form | direct HGB + core soft type | direct HGB + soft type | `0.0891` | `+0.0080` | `26.5` | `0.653` |
| 17 | geometry baseline | `log_footprint_only` | HGB, no type | `0.1171` | `+0.0360` | `33.8` | `0.683` |

Interpretation:

- `log_footprint_m2` is the dominant geometry feature.
- Predicted class identity is the key type signal; confidence-only features are weaker.
- Residual HGB on ridge geometry is better than direct HGB with the same feature family.
- Oracle true type reaches `0.0186` log MAE, showing headroom if type semantics were perfect, but it is non-deployable.

---

### 3.4 M30: Predicted-Type Contribution Micro-Ablation

**Question.** For the best `log_footprint_m2` population variant, how much gain comes from predicted type rather than the residual-HGB architecture?

Result:

| Variant | Model | Type features | Val log MAE | Factor-2 | MAE | R2 |
|---|---|---|---:|---:|---:|---:|
| `log_footprint_only` | residual HGB on ridge geometry | soft predicted 3-class type | `0.0738` | `0.9625` | `14.62` | `0.8464` |
| `log_footprint_only_no_type_residual` | residual HGB on ridge geometry | none | `0.1148` | `0.9426` | `27.20` | `0.7801` |
| `log_footprint_only` | direct HGB geometry | none | `0.1171` | `0.9429` | `33.75` | `0.6825` |

Interpretation:

- Within the same residual-HGB architecture, predicted type improves log MAE by about `0.0410`.
- The residual architecture alone improves the direct no-type baseline by only about `0.0023`.
- Therefore the main deployable gain comes from predicted type, not only from model form.

---

## 4. Historical Compatibility-Head Three-Class Evidence

These experiments are still important because they established the original drop-both pipeline and motivated the native replacement. However, they used a historical five-logit compatibility/projection setup and are no longer the final model contract.

### 4.1 M25: Strict Type Seed Sweep

**Question.** Does the chosen drop-both type recipe remain stable after removing non-deployable type geometry columns?

Strict type recipe:

```text
ConvNeXt-Tiny
RGB + mask + deployable geometry
mask_m pooling
CE baseline
weighted sampler
inverse-sqrt class weights
classes projected to residential_small / residential_multi / commercial
```

M25 seed metrics:

| Seed | Macro-F1 | Accuracy | ECE | Commercial F1 | Residential multi F1 | Residential small F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | `0.8722` | `0.9338` | `0.0535` | `0.9627` | `0.7945` | `0.8593` |
| 123 | `0.8611` | `0.9304` | `0.0574` | `0.9611` | `0.7877` | `0.8346` |
| 2025 | `0.8629` | `0.9310` | `0.0563` | `0.9615` | `0.7867` | `0.8406` |
| 3407 | `0.8649` | `0.9351` | `0.0561` | `0.9643` | `0.7914` | `0.8392` |
| 7777 | `0.8547` | `0.9260` | `0.0597` | `0.9585` | `0.7697` | `0.8358` |
| Mean ± std | `0.8632 ± 0.0057` | `0.9313 ± 0.0030` | about `0.0566` | `0.9616 ± 0.0019` | `0.7860 ± 0.0086` | `0.8419 ± 0.0089` |

Interpretation:

- Compatibility-head type prediction was very stable on fixed `m20_split`.
- Commercial was the easiest retained class.
- Most important errors were later shown to be `commercial ↔ residential_multi`.

### 4.2 M29: Historical Type Input and Weighting Ablation

**Question.** Which type-model inputs and imbalance controls support the strict three-class type model?

M29 main metrics:

| Variant | Macro-F1 | Accuracy | ECE | NLL | Multi F1 | Small F1 | Commercial F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rgb_mask_seed2025` | `0.8681` | `0.9310` | `0.0565` | `0.3810` | `0.7883` | `0.8550` | `0.9609` |
| `m28_5seed_type_ensemble` | `0.8679` | `0.9338` | `0.0373` | `0.2388` | `0.7937` | `0.8467` | `0.9633` |
| `rgb_mask_logfootprint_only_seed2025` | `0.8658` | `0.9310` | `0.0523` | `0.3335` | `0.7900` | `0.8462` | `0.9612` |
| `current_strict_seed2025` | `0.8629` | `0.9310` | `0.0563` | `0.4072` | `0.7867` | `0.8406` | `0.9615` |
| `rgb_mask_geometry_no_logfootprint_seed2025` | `0.8619` | `0.9273` | `0.0573` | `0.3566` | `0.7777` | `0.8489` | `0.9591` |
| `current_no_weighting_seed2025` | `0.8464` | `0.9260` | `0.0568` | `0.3520` | `0.7616` | `0.8189` | `0.9586` |
| `rgb_only_seed2025` | `0.8165` | `0.9260` | `0.0624` | `0.4435` | `0.7694` | `0.7188` | `0.9614` |

Current strict confusion matrix, rows=true and columns=predicted in class order `residential_small`, `residential_multi`, `commercial`:

```text
58 7 0
12 402 99
3 100 2523
```

M28 ensemble confusion matrix:

```text
58 7 0
12 404 97
2 94 2530
```

Interpretation:

- RGB-only is weaker, especially for `residential_small`.
- Mask input is valuable.
- `log_footprint_m2` is not required for type classification; geometry variants cluster in a narrow single-seed band.
- Weighting matters; removing sampler/class weights lowers macro-F1 and hurts `residential_multi` recall.
- The dominant type error is `commercial ↔ residential_multi`, not small-vs-multi.

### 4.3 M26: Strict Type-to-Population Seed Sweep

**Question.** Does the population model remain stable when each seed gets strict OOF type features?

Protocol:

- For each M25 seed, generate five tile-blocked OOF type models on train rows.
- Validation type features come from the matching M25 full-train model.
- Population model: residual HGB on ridge geometry with predicted soft type probabilities.
- Primary population geometry: `log_footprint_m2`.

M26 population seed metrics:

| Seed | Geometry | Val log MAE | Delta vs baseline | Factor-2 | MAE | R2 |
|---:|---|---:|---:|---:|---:|---:|
| 42 | `log_footprint_m2` | `0.0748` | `-0.0436` | `0.9600` | `15.12` | `0.8337` |
| 123 | `log_footprint_m2` | `0.0741` | `-0.0433` | `0.9604` | `14.05` | `0.8569` |
| 2025 | `log_footprint_m2` | `0.0700` | `-0.0469` | `0.9619` | `14.08` | `0.8335` |
| 3407 | `log_footprint_m2` | `0.0690` | `-0.0480` | `0.9625` | `11.78` | `0.8687` |
| 7777 | `log_footprint_m2` | `0.0794` | `-0.0381` | `0.9594` | `16.24` | `0.8234` |
| Mean ± std | `log_footprint_m2` | `0.0735 ± 0.0037` | `-0.0440 ± 0.0035` | `0.9609 ± 0.0012` | `14.25 ± 1.48` | `0.8432 ± 0.0168` |
| Mean ± std | core geometry backup | `0.0789 ± 0.0036` | `-0.0288 ± 0.0039` | `0.9611 ± 0.0020` | `15.68 ± 1.72` | `0.8471 ± 0.0435` |

Interpretation:

- M26 was the main historical compatibility-head population stability result.
- It remains useful as historical evidence, but M32 replaces it as the final native-head stability claim.

### 4.4 M28: Historical Five-Seed Type-Probability Ensemble

**Question.** Does probability averaging across strict type seeds improve same-split type and population results?

M28 metrics:

| Component | Metric | Value |
|---|---|---:|
| Type ensemble | macro-F1 | `0.8679` |
| Type ensemble | accuracy | `0.9338` |
| Type ensemble | ECE | `0.0373` |
| Population ensemble, `log_footprint_m2` | log MAE | `0.0694` |
| Population ensemble, `log_footprint_m2` | factor-2 | `0.9638` |
| Population ensemble, `log_footprint_m2` | MAE | `13.32` |
| Population ensemble, `log_footprint_m2` | R2 | `0.8528` |
| Population ensemble, core geometry backup | log MAE | `0.0786` |
| Population ensemble, core geometry backup | factor-2 | `0.9597` |

Interpretation:

- M28 is the best historical compatibility/projection same-split result.
- It has run count 1 and should be described as an ensemble check, not a seed-stability result.

---

## 5. Final Native Three-Class Replacement: M31-M35

The native3 replacement removes the compatibility-head caveat by training a real three-logit head over the retained classes.

```text
--num_classes 3
--class_names residential_small,residential_multi,commercial
```

### 5.1 M31: Native Type Seed Stability

| Metric | Mean | Std |
|---|---:|---:|
| Accuracy | `0.9300` | `0.0035` |
| Macro-F1 | `0.8529` | `0.0106` |
| ECE | `0.0542` | `0.0042` |
| NLL | `0.3385` | `0.0646` |

Per-class mean metrics:

| Class | Precision mean ± std | Recall mean ± std | F1 mean ± std |
|---|---:|---:|---:|
| `residential_small` | `0.7841 ± 0.0383` | `0.8492 ± 0.0450` | `0.8141 ± 0.0241` |
| `residential_multi` | `0.7856 ± 0.0165` | `0.7809 ± 0.0134` | `0.7831 ± 0.0098` |
| `commercial` | `0.9621 ± 0.0030` | `0.9611 ± 0.0043` | `0.9616 ± 0.0021` |

Interpretation:

- Native type performance is slightly weaker than compatibility-head projection, but it matches the final task definition and removes a reporting caveat.

### 5.2 M32: Native Population Seed Stability

| Geometry | Runs | Log MAE mean ± std | Factor-2 mean ± std | MAE mean ± std | R2 mean ± std |
|---|---:|---:|---:|---:|---:|
| `log_footprint_m2` | 5 | `0.0760 ± 0.0032` | `0.9603 ± 0.0034` | `14.76 ± 0.91` | `0.8743 ± 0.0370` |
| core geometry backup | 5 | `0.0814 ± 0.0012` | `0.9582 ± 0.0008` | `16.55 ± 0.61` | `0.8562 ± 0.0418` |

Interpretation:

- M32 is the **primary native-head fixed-split seed-stability claim**.
- It should be described as stability across training seeds on fixed `m20_split`, not as cross-split robustness.

### 5.3 M33: Native Same-Split Ensemble Check

| Component | Metric | Value |
|---|---|---:|
| Type ensemble | macro-F1 | `0.8579` |
| Type ensemble | accuracy | `0.9332` |
| Type ensemble | ECE | `0.0342` |
| Type ensemble | NLL | `0.2224` |
| Population ensemble, `log_footprint_m2` | log MAE | `0.0734` |
| Population ensemble, `log_footprint_m2` | factor-2 | `0.9607` |

Interpretation:

- M33 improves calibration and recovers population performance on the same split.
- M33 is the packaged inference default but remains a same-split ensemble check, not a multi-run stability result.

### 5.4 M34: Native Type Input Ablation

| Variant | Macro-F1 | Accuracy | ECE | NLL |
|---|---:|---:|---:|---:|
| RGB + mask | `0.8626` | `0.9298` | `0.0537` | `0.3314` |
| strict RGB + mask + core geometry | `0.8601` | `0.9304` | `0.0598` | `0.4430` |
| RGB + mask + `log_footprint_m2` only | `0.8598` | `0.9310` | `0.0534` | `0.3276` |
| M33 native five-seed ensemble | `0.8579` | `0.9332` | `0.0342` | `0.2224` |
| RGB + mask + geometry without `log_footprint_m2` | `0.8495` | `0.9301` | `0.0577` | `0.3934` |
| strict architecture without sampler / class weights | `0.8262` | `0.9282` | `0.0536` | `0.2924` |
| RGB only | `0.8048` | `0.9154` | `0.0586` | `0.2986` |

Native strict seed2025 confusion matrix, rows=true and columns=predicted:

```text
53 10 2
7 399 107
2 95 2529
```

Native M33 ensemble confusion matrix:

```text
56 9 0
14 405 94
2 95 2529
```

Interpretation:

- Mask input remains important.
- RGB-only is clearly weaker.
- Geometry is useful but not the sole driver of type classification.
- Imbalance handling is important.
- Dominant errors remain `residential_multi ↔ commercial`.

### 5.5 M35: Native Population Type-Contribution Check

| Model | Log MAE | Factor-2 | MAE | R2 |
|---|---:|---:|---:|---:|
| `log_footprint_m2 + soft native predicted type + residual HGB` | `0.0742` | `0.9569` | `13.87` | `0.8614` |
| `log_footprint_m2 + no type + residual HGB` | `0.1157` | `0.9438` | `33.71` | `0.6817` |
| `log_footprint_m2 + no type + direct HGB` | `0.1175` | `0.9423` | `33.86` | `0.6782` |

Interpretation:

- Predicted type remains highly valuable under the final native3 contract.
- The exact residual-HGB comparison improves from `0.1157` without type to `0.0742` with soft predicted type.

---

## 6. M41: Native3 Split Sensitivity, Calibration, and Uncertainty

M41 is the final diagnostic block for the selected native3 pipeline.

### 6.1 Split-sensitivity check

**Question.** How much do results move under fresh tile-blocked validation splits?

Type-only split sensitivity:

| Split | Split seed | Type seed | Val rows | Macro-F1 | Accuracy | ECE | Small F1 | Multi F1 | Commercial F1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| m20 | 2025 | 5-seed mean | 3,204 | `0.8529 ± 0.0106` | `0.9300 ± 0.0035` | `0.0542 ± 0.0042` | `0.8141 ± 0.0241` | `0.7831 ± 0.0098` | `0.9616 ± 0.0021` |
| alt1 | 3001 | 2025 | 2,919 | `0.8428` | `0.9058` | `0.0788` | `0.8283` | `0.7565` | `0.9437` |
| alt2 | 3002 | 2025 | 2,878 | `0.8907` | `0.9538` | `0.0362` | `0.8571` | `0.8402` | `0.9748` |

Population split sensitivity:

| Split | Split seed | Type seed | Val log MAE | Factor-2 | MAE | R2 |
|---|---:|---|---:|---:|---:|---:|
| m20 | 2025 | 5-seed mean | `0.0760 ± 0.0032` | `0.9603 ± 0.0034` | `14.76 ± 0.91` | `0.8743 ± 0.0370` |
| alt1 | 3001 | 2025 | `0.1041` | `0.9448` | `24.31` | `0.7261` |
| alt2 | 3002 | 2025 | `0.0613` | `0.9726` | `15.96` | `0.9854` |

Interpretation:

- Cross-split variation is materially larger than same-split seed variation.
- `m20_split` is not an obvious optimistic outlier: alt1 is weaker and alt2 is stronger.
- Population sensitivity follows type sensitivity.
- Report M31/M32 as fixed-split seed stability, not cross-split robustness.

### 6.2 Calibration check

**Question.** Should calibrated type probabilities replace raw ensemble probabilities for downstream population regression?

Native3 M33 calibration check:

| Method | Type macro-F1 | Accuracy | ECE | NLL | Argmax flips | Pop log MAE | Factor-2 | MAE | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw M33 ensemble | `0.8579` | `0.9332` | `0.0342` | `0.2224` | `0` | `0.0734` | `0.9607` | `14.35` | `0.8690` |
| temperature | `0.8579` | `0.9332` | `0.0097` | `0.1738` | `0` | `0.0827` | `0.9585` | `15.99` | `0.8754` |
| vector/classwise | `0.8576` | `0.9332` | `0.0064` | `0.1713` | `4` | `0.0816` | `0.9576` | `15.41` | `0.8829` |

Decision:

- Temperature scaling improves type ECE/NLL and preserves macro-F1.
- Vector/classwise calibration improves ECE/NLL further but introduces four argmax flips.
- Both calibration options worsen downstream population log MAE.
- Therefore raw M33 ensemble probabilities remain the default population input.
- Temperature calibration may be kept only as an optional type-probability diagnostic.

### 6.3 Ensemble uncertainty summary

Native3 M33 ensemble uncertainty summary:

| Metric | Mean | Median | P90 | P95 |
|---|---:|---:|---:|---:|
| `type_pmax` | `0.9673` | `0.9998` | `1.0000` | `1.0000` |
| `type_margin` | `0.9351` | `0.9996` | `1.0000` | `1.0000` |
| `type_entropy` | `0.0759` | `0.0018` | `0.3487` | `0.5852` |
| `type_vote_entropy` | `0.0474` | `0.0000` | `0.0000` | `0.5004` |
| `var_pred_type_prob` | `0.0099` | `0.0000` | `0.0197` | `0.0952` |
| `var_pred_population` | `3820.8185` | `0.0000` | `0.3621` | `83.2015` |
| `std_pred_population` | `6.7915` | `0.0029` | `0.6018` | `9.1202` |

Interpretation:

- Most rows have very high ensemble confidence.
- The uncertainty tail is concentrated in a small subset.
- Top uncertain rows are useful review targets, not evidence of broad uncertainty failure.

Top uncertain examples are written to:

```text
II_package/outputs/stage2a/population_type_features/m41_native3_split_sensitivity/m33_native3_uncertainty/native3_uncertainty_examples.csv
```

---

## 7. Inference, Integration, and Output Status

### 7.1 Current experiment-level outputs

- Type validation inference is produced by `train_stage2a.py --save_val_predictions`.
- Each type run writes `stage2a_best_model.pt` and `best_val_predictions.csv`.
- M26/M28 compatibility-head and M32/M33 native-head population fitting write validation prediction CSVs under each run's `predictions/` directory.
- Selected validation population prediction file name: `predictions/residual_hgb_on_ridge_geometry_pred_type_soft.csv`.
- M33 native ensemble validation type predictions: `type_val_m31_native3_5seed_mean.csv`.

### 7.2 Stage-1 to Stage-2a adapter

`build_stage2a_infer_csv.py` converts Stage-1 shared instance tables into Stage-2a inputs.

Output columns include:

- `building_uid`
- `crop_path`
- `mask_path`
- metadata passthrough fields
- mask-derived geometry features
- `footprint_m2`, `log_footprint_m2`, and `footprint_m2_source` when footprint is derived from mask area

When upstream shared CSVs do not include footprint columns, it derives:

```text
footprint_m2 = mask_area_px * footprint_m2_per_mask_px
```

Default scale:

```text
4.641392
```

This is the Stage-2a train-manifest median `footprint_m2 / mask_area_px`.

### 7.3 End-to-end driver default

The driver now supports the packaged native3 default:

```bash
python3 scripts/driver/run_instance_impact_driver.py \
  --pre_image <PRE_IMAGE.png> \
  --post_image <POST_IMAGE.png> \
  --stage2a_infer_mode native3_ensemble
```

Remote smoke test:

- `infer_stage2a_native3_ensemble.py` completed on 32 rows from `labels_manifest_m20_drop_institutional_other.csv`.
- It loaded all five native checkpoints.
- It reported `temperature_scaling: none`.
- It wrote `outputs/stage2a/smoke/native3_ensemble_infer_smoke32.csv`.

---

## 8. Current Status

- Stage-2a now has a documented final native three-class exposure pipeline.
- The selected taxonomy is closed-world: `residential_small`, `residential_multi`, `commercial`.
- `institutional` and `other` are excluded from the selected contract.
- The final population model is a regression model for `estimated_population` proxy labels.
- The primary native fixed-split metric is M32: log MAE `0.0760 ± 0.0032`.
- The best native same-split ensemble check is M33: log MAE `0.0734`.
- The packaged validation check reproduces the M33 population result: log MAE `0.073351`, factor-2 `0.960674`.
- M41 split sensitivity is complete: alt1 is weaker (`0.1041` log MAE), alt2 is stronger (`0.0613` log MAE).
- Therefore, report fixed-split seed stability separately from cross-split robustness.
- M41 calibration shows raw M33 ensemble probabilities should remain the default for downstream population inference.
- Historical compatibility-head results remain useful evidence, but the final model contract is native three-class.

Remaining optional work, only if broader claims are needed:

- independent held-out test or broader blocked CV;
- stronger handling of `parcel_code` and low-support residential slices;
- explicit all-row exposure accounting for dropped `institutional` / `other` classes.

---

# Appendices

## Appendix A. Non-Selected Label-Policy Search Context

This appendix records non-selected M20 policies. They are useful for context but are not part of the selected Stage2b-facing three-class contract.

| Policy | Active classes | Rows kept | Best recipe | Val macro-F1 | Val accuracy |
|---|---:|---:|---|---:|---:|
| Merge `institutional` -> `other` | 4 | `23,014` | ConvNeXt logit-adjusted CE tau=0.5 | `0.7269` | `0.8729` |
| Drop `institutional` | 4 | `22,382` | ConvNeXt logit-adjusted CE tau=0.5 | `0.7218` | `0.8867` |
| Drop `other` | 4 | `22,060` | ConvNeXt logit-adjusted CE tau=0.5 | `0.7306` | `0.9060` |
| Drop both | 3 | `21,428` | ConvNeXt weighted CE + `mask_m` | `0.8628` | `0.9304` |

Decision:

- Drop-both was selected because it substantially improves macro-F1 and produces a cleaner final three-class task.
- This comes with a coverage limitation: all main metrics are retained-row closed-world metrics.

---

## Appendix B. Supplementary Drop-Other Four-Class Diagnostic: M36-M40

This appendix records the later diagnostic that retained `institutional` by dropping only `other`. It is intentionally excluded from the main experiment body because the selected final contract remains native three-class drop-both.

### B.1 Diagnostic question

If we drop only `other` and retain `institutional`, can a native four-head type-to-population pipeline perform well enough to avoid the drop-both coverage limitation?

Native four-class contract:

```text
--num_classes 4
--class_names residential_small,residential_multi,commercial,institutional
```

Dataset:

| Policy | Active classes | Rows kept | Validation rows |
|---|---|---:|---:|
| `drop_other` | `residential_small`, `residential_multi`, `commercial`, `institutional` | `22,060 / 23,014` | `3,298` |

### B.2 M36 screened institutional-aware type variants

| Variant | Macro-F1 | Accuracy | Institutional F1 | ECE |
|---|---:|---:|---:|---:|
| `tau10_global` | `0.7215` | `0.9027` | `0.3854` | `0.0601` |
| `balanced_global` | `0.7134` | `0.8996` | `0.3654` | `0.0536` |
| `tau075_global` | `0.7127` | `0.9024` | `0.3636` | `0.0681` |
| `tau05_global` | `0.7117` | `0.9015` | `0.3605` | `0.0410` |

### B.3 M37 multi-seed type comparison

| Key | Macro-F1 mean | Accuracy mean | ECE mean | Institutional F1 mean | Institutional precision mean | Institutional recall mean |
|---|---:|---:|---:|---:|---:|---:|
| `tau05_global` | `0.7218` | `0.9046` | `0.0573` | `0.3674` | `0.4259` | `0.3234` |
| `tau10_global` | `0.7202` | `0.9025` | `0.0670` | `0.3800` | `0.3938` | `0.3745` |

### B.4 M39 five-seed type ensemble

| Metric | Value |
|---|---:|
| Macro-F1 | `0.7312` |
| Accuracy | `0.9078` |
| ECE | `0.0449` |
| Institutional precision | `0.4321` |
| Institutional recall | `0.3723` |
| Institutional F1 | `0.4000` |

M39 ensemble confusion matrix, rows=true and columns=`residential_small,residential_multi,commercial,institutional`:

```text
60 3 0 2
17 391 94 11
2 83 2508 33
7 16 36 35
```

### B.5 M38 / M39 / M40 population results

M38 population stability with OOF native four-head type probabilities:

| Geometry | Runs | Log MAE mean ± std | Factor-2 mean ± std | MAE mean ± std | R2 mean ± std |
|---|---:|---:|---:|---:|---:|
| `log_footprint_m2` | 5 | `0.0987 ± 0.0060` | `0.9495 ± 0.0050` | `28.16 ± 1.95` | `0.8946 ± 0.0217` |
| core geometry backup | 5 | `0.1001 ± 0.0019` | `0.9485 ± 0.0014` | `26.39 ± 1.08` | `0.9392 ± 0.0103` |

M39 population with five-seed type ensemble:

| Geometry | Log MAE | Factor-2 | MAE | R2 |
|---|---:|---:|---:|---:|
| `log_footprint_m2` | `0.0909` | `0.9542` | `25.08` | `0.9158` |
| core geometry backup | `0.0928` | `0.9548` | `24.05` | `0.9374` |

M40 type-contribution and ablation highlights:

| Variant | Log MAE | Factor-2 | MAE | R2 |
|---|---:|---:|---:|---:|
| `log_footprint_m2 + soft predicted 4-class type + residual HGB` | `0.0899` | `0.9566` | `22.81` | `0.9352` |
| core geometry + soft predicted 4-class type + residual HGB | `0.0939` | `0.9554` | `24.08` | `0.9393` |
| `log_footprint_m2 + no type + residual HGB` | `0.1303` | `0.9378` | `38.12` | `0.8738` |
| core geometry + no type + residual HGB | `0.1347` | `0.9378` | `37.97` | `0.9050` |
| oracle true type diagnostic | `0.0189` | `0.9985` | `5.28` | `0.9879` |

Diagnostic interpretation:

- Retaining `institutional` recovers only 632 rows relative to drop-both.
- It substantially weakens the type task.
- The limiting class is `institutional`, not `commercial`.
- Predicted type still helps population in the four-class setting.
- The overall four-class pipeline is weaker than the selected native three-class drop-both path.
- Therefore M36-M40 is useful as a limitation analysis, but it does not change the selected Stage2b-facing contract.

---

## Appendix C. Original Stage-2a Clarification and Sanity Run

### C.1 Original Stage-2a implementation

The provided original `stage2a` implementation has two distinct pieces:

1. A rule-based geospatial pipeline (`stage2a/code.ipynb`) that:
   - classifies building type from parcel/land-use + footprint area;
   - estimates units from type;
   - estimates population via `estimated_population = estimated_units * people_per_unit_ratio * occupancy_rate`;
   - applies default ratios and caps.

2. A multi-task vision model (`stage2a/building_population_model.ipynb`) that predicts:
   - building type class logits;
   - `log1p(population)` regression.

Clarification:

- The original rule pipeline does not infer units by reversing model-predicted population.
- The intended rule chain is:

```text
type -> units -> population
```

### C.2 Sanity run on Stage-1 shared assets

Shared artifacts used:

```text
outputs/shared_instances_sanity_2tiles_v2_r48/shared_instance_samples.csv
```

Observed behavior:

- input rows: `114`
- output rows: `114`
- output file: `outputs/shared_instances_sanity_2tiles_v2_r48/stage2a_sanity_predictions.csv`
- pipeline execution and output schema looked correct
- predicted classes and population values were plausible for an out-of-domain sanity pass

Decision:

- For downstream Stage-3 synthesis, use `pred_population` as the primary exposure signal.
- Keep `pred_type_class` as an auxiliary diagnostic field.

---

## Appendix D. Evidence and Verification Index

Primary evidence files:

```text
refine-logs/STAGE2A_TYPE_M20_LABEL_POLICY_RESULTS.md
refine-logs/STAGE2A_M23_DROP_BOTH_POPULATION_RESULTS.md
refine-logs/STAGE2A_M24_DROP_BOTH_POPULATION_ABLATION_RESULTS.md
refine-logs/STAGE2A_M25_M26_DROP_BOTH_STABILITY_PLAN.md
refine-logs/STAGE2A_M29_TYPE_ABLATION_RESULTS.md
refine-logs/STAGE2A_M30_POPULATION_TYPE_CONTRIBUTION_RESULTS.md
refine-logs/STAGE2A_M31_M35_NATIVE3_HEAD_RESULTS.md
refine-logs/STAGE2A_M36_M40_DROP_OTHER_INSTITUTIONAL_RESULTS.md
refine-logs/STAGE2A_M41_NATIVE3_SPLIT_CALIBRATION_UNCERTAINTY_RESULTS.md
II_package/scripts/stage2a/training/package_stage2a_native3_population.py
II_package/scripts/stage2a/infer/infer_stage2a_native3_ensemble.py
refine-logs/stage2a_native3_deploy_packaging.sh
review-stage/AUTO_REVIEW.md
CLAIMS_FROM_RESULTS.md
```

Verification log:

- M25/M26 completed on `tamu-gpu` at `2026-07-07T03:14:05-05:00`.
- M28 completed at `2026-07-07T03:30:46-05:00`.
- M30 completed at `2026-07-07T09:56:12-05:00`.
- M29 completed at `2026-07-07T11:18:12-05:00`.
- M31-M35 completed at `2026-07-07T22:18:00-05:00`.
- M36-M40 completed at `2026-07-08T09:43:09-05:00`.
- M41 completed on `2026-07-09` in screen `stage2a_m41`.
- Native3 deploy packaging completed with validation log MAE `0.073351` and factor-2 `0.960674`.
- Native3 deploy inference smoke test completed on 32 rows with all five checkpoints and no temperature scaling.
- Remote test command passed: `27 passed`.
- M31-M35 remote pre-run tests passed: `21 passed`, and native model-contract sanity returned three logits.
