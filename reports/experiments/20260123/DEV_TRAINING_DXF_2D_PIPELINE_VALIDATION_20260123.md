# DEV_TRAINING_DXF_2D_PIPELINE_VALIDATION_20260123

## Summary
- Ran the local DXF batch analyze pipeline against a 50-file sample of training DXF drawings.
- Results were written to CSV/JSON outputs for follow-up review.

## Observations
- Graph2D model was unavailable in this environment (torch missing), so graph2d fields remained empty.
- Classification still completed via the rule/fusion path.

## Outputs
- `reports/experiments/20260123/dxf_batch_analysis/batch_results.csv`
- `reports/experiments/20260123/dxf_batch_analysis/batch_low_confidence.csv`
- `reports/experiments/20260123/dxf_batch_analysis/summary.json`
- `reports/experiments/20260123/dxf_batch_analysis/label_distribution.csv`
