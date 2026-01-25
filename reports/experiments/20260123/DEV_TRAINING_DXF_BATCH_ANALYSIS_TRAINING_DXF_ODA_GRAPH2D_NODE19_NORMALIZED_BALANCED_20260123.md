# DEV_TRAINING_DXF_BATCH_ANALYSIS_TRAINING_DXF_ODA_GRAPH2D_NODE19_NORMALIZED_BALANCED_20260123

## Summary
- Ran Graph2D batch analysis using the normalized, balanced-sampler node19 model on the ODA-converted DXF set.

## Inputs
- DXF directory: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123`
- Output directory: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_balanced`
- Model: `models/graph2d_training_oda_node19_normalized_balanced_20260123.pth`

## Command
- `GRAPH2D_MODEL_PATH=models/graph2d_training_oda_node19_normalized_balanced_20260123.pth .venv-graph/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123" --output-dir "reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_balanced" --max-files 120`

## Results
- Summary: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_balanced/summary.json`
  - total=110, success=110, error=0
  - confidence buckets: gte_0_8=57, 0_6_0_8=25, 0_4_0_6=28
  - low_confidence_count=53
- CSV: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_balanced/batch_results.csv`
- Low confidence CSV: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_balanced/batch_low_confidence.csv`
- Label distribution: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_balanced/label_distribution.csv`
- Graph2D diff: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_balanced/graph2d_diff.csv`
