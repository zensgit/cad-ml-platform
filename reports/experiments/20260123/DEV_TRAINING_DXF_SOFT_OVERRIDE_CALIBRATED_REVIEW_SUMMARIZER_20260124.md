# DEV_TRAINING_DXF_SOFT_OVERRIDE_CALIBRATED_REVIEW_SUMMARIZER_20260124

## Objective
Automate rollups for the soft-override review template so manual review progress and corrected labels can be tracked with a single command.

## Script
- `scripts/summarize_soft_override_review.py`

## Default Inputs
- Review template: `reports/experiments/20260123/soft_override_calibrated_added_review_template_20260124.csv`

## Default Outputs
- Summary: `reports/experiments/20260123/soft_override_calibrated_added_review_summary_20260124.csv`
- Corrected label counts: `reports/experiments/20260123/soft_override_calibrated_added_correct_label_counts_20260124.csv`

## Command
```
python3 scripts/summarize_soft_override_review.py
```

## Notes
- Supports custom paths via `--review-template`, `--summary-out`, and `--correct-labels-out`.
- Agree rate is computed as agree / reviewed (non-empty `agree_with_graph2d`).
