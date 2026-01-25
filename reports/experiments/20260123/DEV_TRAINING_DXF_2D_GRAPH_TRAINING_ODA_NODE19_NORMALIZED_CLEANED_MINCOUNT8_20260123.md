# DEV_TRAINING_DXF_2D_GRAPH_TRAINING_ODA_NODE19_NORMALIZED_CLEANED_MINCOUNT8_20260123

## Summary
- Trained a Graph2D model on the min-count=8 cleaned normalized label manifest.

## Inputs
- Manifest: `reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_ODA_NORMALIZED_CLEANED_MINCOUNT8_20260123.csv`
- DXF directory: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123`

## Command
- `.venv-graph/bin/python scripts/train_2d_graph.py --manifest "reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_ODA_NORMALIZED_CLEANED_MINCOUNT8_20260123.csv" --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123" --epochs 5 --batch-size 4 --hidden-dim 64 --lr 0.001 --node-dim 19 --output "models/graph2d_training_oda_node19_normalized_cleaned_mincount8_20260123.pth"`

## Output
- `models/graph2d_training_oda_node19_normalized_cleaned_mincount8_20260123.pth`
