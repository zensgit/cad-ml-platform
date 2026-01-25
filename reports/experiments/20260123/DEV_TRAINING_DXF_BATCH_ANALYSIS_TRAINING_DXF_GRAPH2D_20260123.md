# DEV_TRAINING_DXF_BATCH_ANALYSIS_TRAINING_DXF_GRAPH2D_20260123

## Summary
- Ran a 50-file DXF batch analysis from the training DXF directory with Graph2D enabled.

## Inputs
- DXF directory: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf`
- Output directory: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_graph2d`
- Model: `models/graph2d_training_cleaned_20260123.pth`

## Command
- `GRAPH2D_MODEL_PATH=models/graph2d_training_cleaned_20260123.pth .venv-graph/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" --output-dir "reports/experiments/20260123/dxf_batch_analysis_training_dxf_graph2d" --max-files 50`

## Results
- Summary: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_graph2d/summary.json`
  - total=50, success=50, error=0
  - confidence buckets: gte_0_8=22, 0_6_0_8=9, 0_4_0_6=19
  - low_confidence_count=28
- CSV: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_graph2d/batch_results.csv`
- Low confidence CSV: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_graph2d/batch_low_confidence.csv`
- Label distribution: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_graph2d/label_distribution.csv`
