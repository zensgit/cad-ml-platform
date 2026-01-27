# DEV_GRAPH2D_ENSEMBLE_FULL_BATCH_VALIDATION_20260127

## What Was Validated
- Full-batch DXF analysis with `GRAPH2D_ENSEMBLE_ENABLED=true`
- Ensemble metadata presence in batch outputs
- Auto-review and coverage artifacts generation
- Targeted unit tests for filename/title-block helpers

## Validation Commands
```bash
# 1) Auto review
.venv-graph/bin/python scripts/review_soft_override_batch.py \
  --input reports/experiments/20260127/dxf_batch_analysis_training_dxf_random110_ensemble_full_20260127/batch_results.csv \
  --output reports/experiments/20260127/dxf_batch_analysis_training_dxf_random110_ensemble_full_20260127/soft_override_reviewed_20260127.csv \
  --reviewer auto

# 2) Review summaries
.venv-graph/bin/python scripts/summarize_soft_override_review.py \
  --review-template reports/experiments/20260127/dxf_batch_analysis_training_dxf_random110_ensemble_full_20260127/soft_override_reviewed_20260127.csv \
  --summary-out reports/experiments/20260127/dxf_batch_analysis_training_dxf_random110_ensemble_full_20260127/soft_override_review_summary_20260127.csv \
  --correct-labels-out reports/experiments/20260127/dxf_batch_analysis_training_dxf_random110_ensemble_full_20260127/soft_override_correct_label_counts_20260127.csv

# 3) Targeted unit tests
.venv-graph/bin/python -m pytest \
  tests/unit/test_filename_classifier.py \
  tests/unit/test_titleblock_extractor.py -q
```

## Key Validation Results
- Batch summary: 110/110 success, 0 errors
- Ensemble enabled: 110/110
- Drawing-type excluded: 56/110 (50.91%)
- Hybrid source split: filename 107, fusion 3
- Filename coverage: 109/110 (99.09%)
- Auto review: agree 0, disagree 109, unknown 1
- Targeted unit tests: 8 passed

## Validation Artifacts
- `reports/experiments/20260127/dxf_batch_analysis_training_dxf_random110_ensemble_full_20260127/summary.json`
- `reports/experiments/20260127/dxf_batch_analysis_training_dxf_random110_ensemble_full_20260127/ensemble_summary_20260127.csv`
- `reports/experiments/20260127/dxf_batch_analysis_training_dxf_random110_ensemble_full_20260127/soft_override_review_summary_20260127.csv`
- `reports/experiments/20260127/dxf_batch_analysis_training_dxf_random110_ensemble_full_20260127/filename_coverage_summary_20260127.csv`

## Conclusion
The ensemble-enabled pipeline runs cleanly on the full local DXF batch and produces consistent, reviewable artifacts. Filename remains the dominant signal on this dataset.
