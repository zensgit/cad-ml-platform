# DEV_DXF_BASELINE_AUTOREVIEW_VALIDATION_20260125

## Validation summary
- Dataset size: 110 DXF files (cap reached; 300 requested)
- Batch run: 110 success, 0 errors
- Filename coverage: 109/110 matched labels (99.09%)
- Auto-review agreement (Graph2D): 0 agree, 109 disagree, 1 unknown
- Conflicts list: 110 entries (all rows flagged due to missing Graph2D output)

## Evidence
- Batch summary: `reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_baseline_20260125/summary.json`
  - `total=110`, `success=110`, `error=0`
  - Confidence buckets: `0.4–0.6: 28`, `0.6–0.8: 25`, `>=0.8: 57`
  - Low-confidence count: `53`
- Review summary: `reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_baseline_20260125/soft_override_review_summary_20260125.csv`
- Filename coverage: `reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_baseline_20260125/filename_coverage_summary_20260125.csv`
- Conflicts list: `reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_baseline_20260125/soft_override_conflicts_20260125.csv`

## Environment caveats
- Graph2D predictions were unavailable in this run:
  - `Torch not found. 2D vision module disabled.`
  - `Graph2D classifier not available: name 'DXF_NODE_DIM' is not defined`
- As a result, Graph2D agreement/conflict metrics are not representative; this validation focuses on filename-based coverage and artifact integrity.

## Follow-ups
- Re-run the same batch after enabling Torch + Graph2D to obtain meaningful agreement/conflict statistics.
