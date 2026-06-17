# Stage 2a Revision Plan (Exposure: Building Type and Population)

**Status:** Planning document. Stage 2a is the one IIF stage not yet defensible for journal
submission. Stage 2b (`Stage2b.md`) is a fully evidenced stage; this plan brings Stage 2a up
to the same evidentiary bar.

**Why this exists:** `stage2a.md` today is integration notes only, a 114-row out-of-domain
sanity run with no held-out metrics. The checkpoint is bundled in `models/stage2a/`, and the
training data and code are provided in the separately uploaded Stage 2a folder, but the stage
has never been quantitatively evaluated. The exposure half of the framework's decision-support
claim (each building carries an exposure estimate a decision maker can act on) is therefore
unsupported until Stage 2a is measured. The fix is bounded: a proper evaluation, a documented training
protocol, and likely some external validation, all mirroring what Stage 2b already does.

**Confirmed scope decisions:** full Stage 2b parity; ground-truth and external-validation
scope deferred until the input dataset is inspected (see Section C); this document lives in
`II_package/docs/` alongside the other canonical stage docs.

**Orientation:** for the pipeline end to end, read `final_present_instance.ipynb` first. This
document is the forward plan, not a pipeline description; the notebook and `docs/` carry the
detail.

**Source of truth and known issues:**
- `II_package/docs/` is authoritative. If the standalone report PDF travels with this bundle,
  treat it as outdated (it predates SAM3 and the current Stage 2b numbers) and do not cite it.
- The shared-asset step skips polygons parsed as MULTIPOLYGON, a known recall bias of roughly
  8% of instances. This is a limitation with a clear fix (split each MULTIPOLYGON into
  connected components, one row per component), not an intentional quality gate.

---

## Section A. Materials to read first (the whole picture)

Read in this order. Each item is annotated with what it gives you.

1. **`II_package/docs/stage2a.md`** — current Stage 2a status. The two-piece implementation
   (a rule-based geospatial pipeline and a multi-task vision model), the model contract
   (EfficientNet-B0, 4-channel input = RGB + binary `mask_M`, heads for building-type
   classification and `log1p(population)` regression), the 114-row sanity run, and the
   `build_stage2a_infer_csv.py` / `infer_stage2a.py` CLI contract.
2. **`II_package/docs/Stage2b.md`** — the gold-standard methodology to mirror. Index build,
   crop/mask prep, audit, tile-blocked split, multi-seed sweeps, pooling and ring-radius
   ablations, three-method calibration, weighted ensemble, per-sample uncertainty,
   visualization. Every number traces to a recorded command. This is the template Stage 2a
   must match.
3. **`II_package/docs/stage1.md`** and **`II_package/docs/README.md`** — the shared-asset
   contract that Stage 2a consumes: `shared_instance_samples.csv` with `pre_crop`, `mask_M`,
   `mask_R`, plus the stage interfaces and instance-identifier join.
4. **`II_package/README.md`** — runtime package layout and driver flags, including the bundled
   checkpoint path `models/stage2a/stage2a_best_model.pt`.
5. **`II_package/final_present_instance.ipynb`** — end-to-end narrative. Inspect the Stage 2a
   Summary cells (around lines 1108-1147), which currently present exposure qualitatively
   ("differentiated estimates", population-proxy quantiles) with no accuracy metrics.
6. **Exposure target definition** — see `stage2a.md` (item 1) for the `type -> units ->
   population` StratMap proxy. A fuller narrative exists in a separately maintained chapter
   held privately by the author and is not required to execute this plan.
7. **Stage 2a training data and code** in the separately uploaded Stage 2a folder, including
   the rule pipeline (`code.ipynb`) and the multi-task vision model
   (`building_population_model.ipynb`). These define the current architecture, targets, and the
   training data already used.
8. **`II_package/scripts/build_stage2a_infer_csv.py`** and **`II_package/scripts/infer_stage2a.py`**
   — the current inference wiring and the exact prediction I/O schema
   (`pred_population`, `pred_log1p_population`, `pred_type_idx`, `pred_type_class`,
   `pred_type_conf`, class probabilities).
9. **`II_package/configs/stage2b/*.json`** — the shape a locked, citable Stage 2a train config
   should imitate (`run019_seed2025_train_config.json`, `seed7777_*`, `seed9999_*`).
10. **StratMap parcel source and proxy-rule definition** — ground-truth provenance for the
    exposure targets (per-parcel building type and unit count; `type -> units -> population`).

---

## Section B. Revision plan (phased checklist, Stage 2b parity)

### Phase 0 — Recover and inventory (prerequisite)
- [ ] Review the bundled Stage 2a training code, checkpoint, and training data; confirm
      architecture, input channels, target schema, and the data actually used.
- [ ] Inventory the StratMap-derived label set: row count, building-type class balance,
      population distribution, missing-value rate, and the exact `type -> units -> population`
      computation.
- [ ] Record the Phase 0 findings here, then resolve the open decision in Section C.

### Phase 1 — Dataset and split protocol (mirror `build_stage2_index` / `audit`)
- [ ] Build a Stage 2a index linking each instance to its crop, `mask_M`, and ground-truth
      (type, units, population), reusing the shared-asset table.
- [ ] Define a geography- or tile-blocked train/val/test split to prevent spatial leakage.
      Stage 2b used a tile split (seed 42); exposure labels are spatially autocorrelated, so a
      random split would inflate metrics. This is the single most important methodological
      choice for credible Stage 2a numbers.
- [ ] Add an audit step: file integrity, mask-area sanity, label distributions, and an
      explicit train/test leakage check.

### Phase 2 — Training and multi-seed stability
- [ ] Re-train the multi-task EfficientNet-B0 with a documented protocol (pretrained, early
      stopping, EMA, fixed seeds), mirroring Stage 2b conventions.
- [ ] Define and record the multi-task loss: cross-entropy (type) plus Huber or MSE on
      `log1p(population)`, with the head-weighting stated explicitly.
- [ ] Run multiple seeds (target 5, as Stage 2b did); report mean and standard deviation, and
      save validation predictions for calibration and error analysis.

### Phase 3 — Ablations (at least the load-bearing ones)
- [ ] 4-channel (RGB + `mask_M`) vs RGB-only: justify the mask channel as the architectural
      choice (the analogue of Stage 2b's pooling ablation).
- [ ] Multi-task vs single-task heads: test whether joint training helps either target.
- [ ] Learned vision model vs the rule-based ACS/area baseline: a comparison table that
      substantiates demoting the rule pipeline to a diagnostic baseline.

### Phase 4 — Metrics and error characterization (the missing core)
- [ ] Type classification: accuracy, macro-F1, per-class F1, confusion matrix.
- [ ] Population regression: MAE, RMSE, R^2, and a within-factor-of-2 hit rate, on both log
      and linear scales; residuals broken down by building type and by population magnitude.
- [ ] Calibration: temperature scaling on the type head with ECE and NLL (mirror Stage 2b);
      for population, prediction-interval coverage or at least empirical error bars.
- [ ] Lock a Stage 2a best-config results table analogous to Stage 2b's best-config table.

### Phase 5 — Ensemble and per-instance uncertainty (parity)
- [ ] Build a multi-seed weighted ensemble for the exposure heads; emit per-instance exposure
      uncertainty (type entropy, population variance across seeds) so the synthesis layer's
      per-stage uncertainty claim holds for Stage 2a as it does for Stage 2b.

### Phase 6 — External validation (scope per the Section C decision)
- [ ] Run an out-of-region or NSI-based test, characterized honestly as a transfer test:
      wiring and geometry transfer, with accuracy bounded by training scope (mirror the LA
      Fire framing used for Stage 2b).

### Phase 7 — Integration, docs, and manuscript
- [ ] Update the bundled `models/stage2a/` checkpoint with the retrained model; confirm the
      driver default path `models/stage2a/stage2a_best_model.pt` resolves.
- [ ] Promote `stage2a.md` from integration notes to a results logbook mirroring `Stage2b.md`
      (commands, tables, decisions, every number traceable to a command).
- [ ] Update the notebook Stage 2a Summary cells to surface real metrics in place of the
      qualitative "looks differentiated" language.
- [ ] Hand the finalized Stage 2a numbers back to the author for the separately maintained
      chapter and manuscript abstract; at that point the exposure half of the decision-support
      claim becomes supportable.

---

## Section C. Open decisions to resolve in this document
- **Ground-truth source and external-validation scope** (deferred). Options under
  consideration: StratMap-only (Texas) for v1 with NSI named as future external; or integrate
  the USACE National Structure Inventory (NSI) now for an out-of-Texas national test. Decide
  after the Phase 0 dataset inventory and record the choice here.
- **Whether the Phase 5 ensemble is required for v1** or can be deferred, depending on how
  strong the single-model metrics from Phase 4 turn out to be.

---

## Section D. Verification (definition of done)
- Every Stage 2a number destined for the manuscript is reproducible from a committed config
  and a recorded command, exactly as `Stage2b.md` achieves.
- The end-to-end driver and the presentation notebook run, and Stage 2a now carries validated,
  documented metrics rather than a sanity-only output.
- A held-out test table exists for both heads (type and population), with calibration and
  error characterization, and, per the Section C decision, an external or transfer result.
- the bundled `models/stage2a/` checkpoint matches the documented training run, and
  `stage2a.md` reads as a logbook peer of `Stage2b.md`.
