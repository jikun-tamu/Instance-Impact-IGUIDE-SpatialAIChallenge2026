# LA Fire 2025 — Building Damage Assessment (M2b)
**Generated**: 2026-04-12  |  **Method**: M2b — Coverage-aware majority vote  |  **Model**: No retraining

## Dataset Overview

| Metric | Value |
|---|---|
| Cells processed (with buildings) | 120 of 295 |
| Cells with 0 Stage-1 detections | 175 |
| Total building instances | 21,797 |
| Post-disaster dates evaluated | 5 (2025-01-10, 01-14, 01-15, 01-18, 01-19) |

## M2b Damage Distribution (Authoritative)

| Damage Class | Count | % |
|---|---|---|
| No damage (0) | 16,246 | 74.5% |
| Minor (1) | 4,174 | 19.1% |
| Major (2) | 39 | 0.2% |
| Destroyed (3) | 93 | 0.4% |
| Unknown / no valid imagery (-1) | 1,245 | 5.7% |

## Top Cells by M2b Damage (minor + major)

| Cell | Buildings | Minor | Major | % Damaged |
|---|---|---|---|---|
| cell_00524 | 391 | 236 | 3 | 61% |
| cell_00495 | 401 | 202 | 2 | 51% |
| cell_00516 | 379 | 174 | 1 | 46% |
| cell_00507 | 365 | 152 | 1 | 42% |
| cell_00506 | 531 | 169 | 0 | 32% |

## Method Notes

- **M2b = coverage-aware majority vote**: valid date = tile_quality_ok AND crop_quality_ok
- Buildings with 0 valid post-disaster dates → Unknown (not assessed)
- Stage-2b ensemble trained on xBD flood events; applied zero-shot to wildfire (domain mismatch)
- Do NOT use M1 labels — they contain artifacts from nodata satellite strips (1,516 false "destroyed")
