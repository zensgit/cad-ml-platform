# DEV_TRAINING_DXF_BATCH_ANALYSIS_TRAINING_DRAWINGS_20260123

## Summary
- Ran local DXF batch analysis on the training drawings directory.
- Observed only two `.dxf` files available in the provided path.

## Inputs
- DXF directory: `/Users/huazhou/Downloads/训练图纸/训练图纸`
- Output directory: `reports/experiments/20260123/dxf_batch_analysis_training_drawings`

## Command
- `GRAPH2D_MODEL_PATH=models/graph2d_training_cleaned_20260123.pth python3 scripts/batch_analyze_dxf_local.py --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸" --output-dir "reports/experiments/20260123/dxf_batch_analysis_training_drawings" --max-files 50`

## Results
- Summary: `reports/experiments/20260123/dxf_batch_analysis_training_drawings/summary.json`
  - total=2, success=2, error=0
  - label_counts: 盖=1, 其他=1
- CSV: `reports/experiments/20260123/dxf_batch_analysis_training_drawings/batch_results.csv`
- Label distribution: `reports/experiments/20260123/dxf_batch_analysis_training_drawings/label_distribution.csv`

## Notes
- `torch` is not available in this environment, so Graph2D inference was disabled and graph2d fields are empty.
