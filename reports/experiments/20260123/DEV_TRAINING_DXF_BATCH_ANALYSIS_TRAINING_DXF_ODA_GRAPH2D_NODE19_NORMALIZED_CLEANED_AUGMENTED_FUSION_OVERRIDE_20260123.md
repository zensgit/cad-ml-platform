# DEV_TRAINING_DXF_BATCH_ANALYSIS_TRAINING_DXF_ODA_GRAPH2D_NODE19_NORMALIZED_CLEANED_AUGMENTED_FUSION_OVERRIDE_20260123

## Summary
- Ran batch analysis with FusionAnalyzer override enabled (min_conf=0.5).

## Inputs
- DXF directory: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123`
- Output directory: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_fusion_override`
- Model: `models/graph2d_training_oda_node19_normalized_cleaned_augmented_20260123.pth`
- Overrides: `FUSION_ANALYZER_OVERRIDE=true`, `FUSION_ANALYZER_OVERRIDE_MIN_CONF=0.5`

## Command
- `FUSION_ANALYZER_OVERRIDE=true FUSION_ANALYZER_OVERRIDE_MIN_CONF=0.5 GRAPH2D_MODEL_PATH=models/graph2d_training_oda_node19_normalized_cleaned_augmented_20260123.pth .venv-graph/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123" --output-dir "reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_fusion_override" --max-files 120`

## Results
- Summary: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_fusion_override/summary.json`
- CSV: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_fusion_override/batch_results.csv`
- Low confidence CSV: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_fusion_override/batch_low_confidence.csv`
- Label distribution: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_fusion_override/label_distribution.csv`
- Diff CSV: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_fusion_override/fusion_diff.csv`
