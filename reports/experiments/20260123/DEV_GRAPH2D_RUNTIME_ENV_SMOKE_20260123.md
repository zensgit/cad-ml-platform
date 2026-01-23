# DEV_GRAPH2D_RUNTIME_ENV_SMOKE_20260123

## Summary
- Ran a 20-file DXF batch analysis using the local `.env` recommended Graph2D runtime settings.
- Graph2D predictions were present for all samples; 2 samples matched the allowlist and passed the 0.7 confidence gate.

## Outputs
- `reports/experiments/20260123/dxf_batch_analysis_graph2d_runtime_env/batch_results.csv`
- `reports/experiments/20260123/dxf_batch_analysis_graph2d_runtime_env/batch_low_confidence.csv`
- `reports/experiments/20260123/dxf_batch_analysis_graph2d_runtime_env/summary.json`
- `reports/experiments/20260123/dxf_batch_analysis_graph2d_runtime_env/label_distribution.csv`

## Notes
- The batch output captures `graph2d_label` and `graph2d_confidence`; allowlist gating was inferred from these fields.
