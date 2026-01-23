# DEV_TRAINING_DXF_2D_GRAPH_INFERENCE_MINCONF_COMPARE_20260123

## Summary
- Compared Graph2D confidence gating at 0.5 vs 0.6 using the cleaned Graph2D checkpoint.
- Higher threshold reduced the number of predictions passing the gate while average confidence stayed stable.

## Results
- min_conf=0.5: 42/50 predictions passed (avg confidence 0.7265)
- min_conf=0.6: 39/50 predictions passed (avg confidence 0.7265)

## Outputs
- `reports/experiments/20260123/dxf_batch_analysis_graph2d_cleaned_minconf05/`
- `reports/experiments/20260123/dxf_batch_analysis_graph2d_cleaned_minconf06/`
