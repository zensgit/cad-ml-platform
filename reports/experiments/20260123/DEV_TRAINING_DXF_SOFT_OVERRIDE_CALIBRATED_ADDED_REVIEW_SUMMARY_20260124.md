# DEV_TRAINING_DXF_SOFT_OVERRIDE_CALIBRATED_ADDED_REVIEW_SUMMARY_20260124

## Objective
Automate summary tracking for the manual review of the 15 newly eligible soft-override candidates introduced by Graph2D temperature calibration.

## Inputs
- Review template: `reports/experiments/20260123/soft_override_calibrated_added_review_template_20260124.csv`

## Outputs
- Review summary: `reports/experiments/20260123/soft_override_calibrated_added_review_summary_20260124.csv`
- Corrected label distribution: `reports/experiments/20260123/soft_override_calibrated_added_correct_label_counts_20260124.csv`

## Summary Snapshot
- Total candidates: 15
- Reviewed: 15
- Agree with Graph2D: 0
- Disagree with Graph2D: 15
- Unknown: 0
- Agree rate: 0.0000
- Suggested min conf: 0.19

## Notes
- Review results indicate the 0.17–0.18 band produced 0% precision for soft overrides.
- Use the decision template to capture any threshold changes and re-run batch analysis after updates.
