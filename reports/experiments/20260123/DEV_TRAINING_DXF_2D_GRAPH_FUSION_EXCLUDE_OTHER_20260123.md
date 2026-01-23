# DEV_TRAINING_DXF_2D_GRAPH_FUSION_EXCLUDE_OTHER_20260123

## Summary
- Added `GRAPH2D_EXCLUDE_LABELS` to prevent excluded Graph2D labels (default: `other`) from participating in fusion.
- With `other` excluded, override changes drop from 35/50 to 2/50 samples.

## Results
- Changes vs no-override baseline: 2/50
- Changed labels: `再沸器` (2)

## Artifacts
- `reports/experiments/20260123/dxf_batch_analysis_graph2d_fusion_exclude_other/*`
