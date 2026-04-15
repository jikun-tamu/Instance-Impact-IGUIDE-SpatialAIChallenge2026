# SAM3 Building Segmentation — Evaluation Report

**IoU threshold**: 0.5 (detection match criterion)

**Metrics definition**: TP = predicted building matched to a GT building (IoU ≥ threshold); FP = unmatched prediction; FN = unmatched GT building.

---

## Split: TRAIN

| Metric | Value |
|--------|-------|
| Images evaluated | 2799 |
| GT buildings | 162787 |
| Predicted buildings | 71420 |
| Avg pred / image | 25.5 |
| Avg GT / image | 58.2 |
| True Positives (TP) | 47993 |
| False Positives (FP) | 23427 |
| False Negatives (FN) | 114794 |
| **Precision** | **0.6720** |
| **Recall** | **0.2948** |
| **F1** | **0.4098** |
| **Mean IoU (matched pairs)** | **0.7562** |
| Matched pairs | 47993 |

### Per-Disaster Breakdown

| Disaster | Images | GT | Pred | Precision | Recall | F1 | Mean IoU |
|----------|-------:|---:|-----:|----------:|-------:|---:|---------:|
| guatemala-volcano | 18 | 856 | 469 | 0.774 | 0.424 | 0.548 | 0.759 |
| hurricane-florence | 319 | 6446 | 6040 | 0.779 | 0.730 | 0.754 | 0.772 |
| hurricane-harvey | 319 | 23014 | 13174 | 0.651 | 0.373 | 0.474 | 0.785 |
| hurricane-matthew | 238 | 13939 | 3685 | 0.744 | 0.197 | 0.311 | 0.706 |
| hurricane-michael | 343 | 22686 | 17826 | 0.712 | 0.560 | 0.627 | 0.733 |
| mexico-earthquake | 121 | 32271 | 2956 | 0.445 | 0.041 | 0.075 | 0.779 |
| midwest-flooding | 279 | 8756 | 4595 | 0.631 | 0.331 | 0.434 | 0.731 |
| palu-tsunami | 113 | 31394 | 3605 | 0.774 | 0.089 | 0.159 | 0.778 |
| santa-rosa-wildfire | 226 | 12950 | 9187 | 0.659 | 0.468 | 0.547 | 0.777 |
| socal-fire | 823 | 10475 | 9883 | 0.592 | 0.558 | 0.575 | 0.751 |
