# DEV_TRAINING_DXF_BATCH_ANALYSIS_TRAINING_DRAWINGS_GRAPH2D_20260123

## Summary
- Re-ran local DXF batch analysis using the Graph2D model with torch-enabled environment.

## Inputs
- DXF directory: `/Users/huazhou/Downloads/训练图纸/训练图纸`
- Output directory: `reports/experiments/20260123/dxf_batch_analysis_training_drawings_graph2d`
- Model: `models/graph2d_training_cleaned_20260123.pth`

## Command
- `GRAPH2D_MODEL_PATH=models/graph2d_training_cleaned_20260123.pth .venv-graph/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸" --output-dir "reports/experiments/20260123/dxf_batch_analysis_training_drawings_graph2d" --max-files 50`

## Results
- Summary: `reports/experiments/20260123/dxf_batch_analysis_training_drawings_graph2d/summary.json`
  - total=2, success=2, error=0
  - label_counts: 盖=1, 其他=1
- CSV: `reports/experiments/20260123/dxf_batch_analysis_training_drawings_graph2d/batch_results.csv`
- Label distribution: `reports/experiments/20260123/dxf_batch_analysis_training_drawings_graph2d/label_distribution.csv`

## Notes
- Graph2D predictions were present; one file hit label `再沸器` at 0.948 confidence.
