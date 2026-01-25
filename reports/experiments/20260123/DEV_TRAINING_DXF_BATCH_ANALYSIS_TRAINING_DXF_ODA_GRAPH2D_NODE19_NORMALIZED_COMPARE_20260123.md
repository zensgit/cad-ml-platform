# DEV_TRAINING_DXF_BATCH_ANALYSIS_TRAINING_DXF_ODA_GRAPH2D_NODE19_NORMALIZED_COMPARE_20260123

## Summary
- Compared the node19 baseline run against the normalized node19 model on the ODA DXF set.

## Inputs
- Baseline summary: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19/summary.json`
- Normalized summary: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized/summary.json`
- Diff CSV: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized/graph2d_diff.csv`

## Findings
- Aggregate metrics unchanged (total=110, low_confidence_count=53, confidence buckets identical).
- Graph2D confidence values changed for all files (retrained model).
- Graph2D label changed for 110 files (normalized model heavily predicts 传动件).
