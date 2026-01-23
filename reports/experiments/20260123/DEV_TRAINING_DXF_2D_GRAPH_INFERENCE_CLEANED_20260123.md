# DEV_TRAINING_DXF_2D_GRAPH_INFERENCE_CLEANED_20260123

## Summary
- Ran DXF batch analysis with the cleaned Graph2D model and confidence gating enabled.
- Graph2D predictions now show higher average confidence across the 50-file sample.

## Outputs
- `reports/experiments/20260123/dxf_batch_analysis_graph2d_cleaned_training/batch_results.csv`
- `reports/experiments/20260123/dxf_batch_analysis_graph2d_cleaned_training/batch_low_confidence.csv`
- `reports/experiments/20260123/dxf_batch_analysis_graph2d_cleaned_training/summary.json`
- `reports/experiments/20260123/dxf_batch_analysis_graph2d_cleaned_training/label_distribution.csv`

## Notes
- `GRAPH2D_MIN_CONF=0.4` gated low-confidence predictions from feeding fusion.
- 44/50 Graph2D predictions exceeded the 0.4 threshold.
