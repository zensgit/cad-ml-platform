# DEV_TRAINING_DXF_BATCH_ANALYSIS_TRAINING_DXF_ODA_GRAPH2D_NODE19_NORMALIZED_BALANCED_COMPARE_20260123

## Summary
- Compared the normalized node19 Graph2D run against the balanced-sampler model.

## Inputs
- Baseline summary: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized/summary.json`
- Balanced summary: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_balanced/summary.json`
- Diff CSV: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_balanced/graph2d_diff.csv`

## Findings
- Aggregate metrics unchanged (total=110, low_confidence_count=53, confidence buckets identical).
- Graph2D labels changed for 110 files (balanced model predicts 开孔件 for most files).
- Graph2D confidences changed for 110 files.
