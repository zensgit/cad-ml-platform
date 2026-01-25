# DEV_DXF_BASELINE_AUTOREVIEW_GRAPH2D_VALIDATION_20260125

## Validation summary
- Dataset size: 110 DXF files (cap reached; 300 requested)
- Batch run: 110 success, 0 errors
- Filename coverage: 109/110 matched labels (99.09%)
- Auto-review agreement (Graph2D vs filename): 0 agree, 109 disagree, 1 unknown
- Conflicts list: 110 entries

## Evidence
- Batch summary: `reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_graph2d_20260125/summary.json`
  - `total=110`, `success=110`, `error=0`
  - Confidence buckets: `0.4–0.6: 28`, `0.6–0.8: 25`, `>=0.8: 57`
  - Low-confidence count: `53`
- Review summary: `reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_graph2d_20260125/soft_override_review_summary_20260125.csv`
- Filename coverage: `reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_graph2d_20260125/filename_coverage_summary_20260125.csv`
- Conflicts list: `reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_graph2d_20260125/soft_override_conflicts_20260125.csv`

## Environment notes
- Graph2D enabled via `.venv-graph` (Torch + ezdxf available).
- Graph2D predictions are not aligned with filename-derived labels in this dataset.

## Follow-ups
- Validate Graph2D against a manually curated golden set (20–50 samples) to confirm if label mismatch stems from label taxonomy or model drift.
