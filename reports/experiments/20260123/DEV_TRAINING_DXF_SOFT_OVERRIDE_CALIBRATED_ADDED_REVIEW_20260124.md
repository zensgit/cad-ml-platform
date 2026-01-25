# DEV_TRAINING_DXF_SOFT_OVERRIDE_CALIBRATED_ADDED_REVIEW_20260124

## Objective
Generate a manual review checklist for the 15 newly eligible soft-override candidates introduced by Graph2D temperature calibration.

## Inputs
- Added candidates: `reports/experiments/20260123/soft_override_calibrated_added_candidates_20260124.csv`

## Outputs
- Review template: `reports/experiments/20260123/soft_override_calibrated_added_review_template_20260124.csv`
- Label distribution: `reports/experiments/20260123/soft_override_calibrated_added_label_counts_20260124.csv`
- Confidence buckets: `reports/experiments/20260123/soft_override_calibrated_added_confidence_buckets_20260124.csv`

## Summary
- Added candidates: 15
- Graph2D labels:
  - 传动件: 15
- Graph2D confidence buckets:
  - 0.17–0.18: 15

## Review Fields Added
- reviewer
- review_date
- agree_with_graph2d
- correct_label
- notes

## Notes
- All newly eligible candidates cluster just above the 0.17 threshold; manual review is recommended before adjusting thresholds.
