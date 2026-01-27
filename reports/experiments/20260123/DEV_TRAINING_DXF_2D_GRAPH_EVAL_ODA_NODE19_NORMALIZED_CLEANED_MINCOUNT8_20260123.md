# DEV_TRAINING_DXF_2D_GRAPH_EVAL_ODA_NODE19_NORMALIZED_CLEANED_MINCOUNT8_20260123

## Summary
- Evaluated the min-count=8 cleaned Graph2D model on a validation split of the ODA DXF dataset.

## Command
- `.venv-graph/bin/python scripts/eval_2d_graph.py --manifest "reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_ODA_NORMALIZED_CLEANED_MINCOUNT8_20260123.csv" --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123" --checkpoint "models/graph2d_training_oda_node19_normalized_cleaned_mincount8_20260123.pth" --batch-size 4 --seed 42 --val-split 0.2 --output-metrics "reports/experiments/20260123/MECH_TRAINING_DXF_GRAPH2D_NODE19_NORMALIZED_CLEANED_MINCOUNT8_METRICS_20260123.csv" --output-errors "reports/experiments/20260123/MECH_TRAINING_DXF_GRAPH2D_NODE19_NORMALIZED_CLEANED_MINCOUNT8_ERRORS_20260123.csv"`

## Outputs
- `reports/experiments/20260123/MECH_TRAINING_DXF_GRAPH2D_NODE19_NORMALIZED_CLEANED_MINCOUNT8_METRICS_20260123.csv`
- `reports/experiments/20260123/MECH_TRAINING_DXF_GRAPH2D_NODE19_NORMALIZED_CLEANED_MINCOUNT8_ERRORS_20260123.csv`
