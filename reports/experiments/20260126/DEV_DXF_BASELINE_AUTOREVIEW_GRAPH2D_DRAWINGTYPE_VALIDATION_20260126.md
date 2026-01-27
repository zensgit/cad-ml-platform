# DEV_DXF_BASELINE_AUTOREVIEW_GRAPH2D_DRAWINGTYPE_VALIDATION_20260126

## Validation summary
- Dataset size: 110 DXF files (cap reached; 300 requested)
- Batch run: 110 success, 0 errors
- Filename coverage: 109/110 matched labels (99.09%)
- Graph2D drawing-type labels excluded from override eligibility
- Soft-override candidates: 0 (expected with drawing-type exclusion)

## Evidence
- Batch summary: `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_drawingtype_20260126/summary.json`
  - `total=110`, `success=110`, `error=0`
  - Confidence buckets: `0.4–0.6: 28`, `0.6–0.8: 25`, `>=0.8: 57`
  - Low-confidence count: `53`
  - Soft-override candidates: `0`
- Review summary: `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_drawingtype_20260126/soft_override_review_summary_20260126.csv`
- Filename coverage: `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_drawingtype_20260126/filename_coverage_summary_20260126.csv`
- Conflicts list: `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_drawingtype_20260126/soft_override_conflicts_20260126.csv`

## Environment notes
- Graph2D enabled via `.venv-graph` (Torch + ezdxf available).
- Drawing-type labels are excluded from fusion and override eligibility in this run.
