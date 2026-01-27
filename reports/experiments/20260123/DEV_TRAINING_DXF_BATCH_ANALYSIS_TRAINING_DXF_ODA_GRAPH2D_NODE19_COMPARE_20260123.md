# DEV_TRAINING_DXF_BATCH_ANALYSIS_TRAINING_DXF_ODA_GRAPH2D_NODE19_COMPARE_20260123

## Summary
- Compared the baseline ODA Graph2D run (node_dim=9 model) against the node19 model.

## Inputs
- Baseline summary: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d/summary.json`
- Node19 summary: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19/summary.json`
- Diff CSV: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19/graph2d_diff.csv`

## Findings
- Aggregate metrics unchanged (total=110, low_confidence_count=53, confidence buckets identical).
- Graph2D confidence values changed for all files (model re-trained).
- Graph2D label changed for 3 files (see diff CSV).
