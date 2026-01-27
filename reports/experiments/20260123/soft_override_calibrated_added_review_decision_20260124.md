# Soft Override Calibrated Added Review Decision (2026-01-24)

## Context
- Review template: `reports/experiments/20260123/soft_override_calibrated_added_review_template_20260124.csv`
- Summary tracker: `reports/experiments/20260123/soft_override_calibrated_added_review_summary_20260124.csv`
- Corrected labels: `reports/experiments/20260123/soft_override_calibrated_added_correct_label_counts_20260124.csv`

## Summary Snapshot
- Total candidates: 15
- Reviewed: 15
- Agree with Graph2D: 0
- Disagree with Graph2D: 15
- Unknown: 0
- Agree rate: 0.0000

## Decision
- Proposed `GRAPH2D_SOFT_OVERRIDE_MIN_CONF`: 0.19
- Rationale:
  - All newly eligible candidates in the 0.17–0.18 band were incorrect.
  - Raising the threshold to 0.19 removes the low-confidence band that produced 0% precision.
  - Additional review is required for any candidates >= 0.19 before enabling broader soft overrides.

## Observations
- Common mislabels: Graph2D predicted `传动件` for all 15 samples.
- Correct labels span 8 classes (人孔/出料凸缘/捕集器组件/拖轮组件/自动进料装置/调节螺栓/捕集口/真空组件).
- Notes: All reviewed samples clustered at ~0.17 confidence; treat this region as unreliable.

## Follow-ups
- [ ] Apply threshold update in runtime config
- [ ] Re-run batch analysis with new threshold
- [ ] Update documentation/verification log
