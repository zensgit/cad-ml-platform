# DEV_TRAINING_DXF_2D_GRAPH_TRAINING_ODA_NODE19_NORMALIZED_BALANCED_20260123

## Summary
- Trained a Graph2D model with balanced sampling on the normalized ODA DXF labels.

## Inputs
- Manifest: `reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_ODA_NORMALIZED_20260123.csv`
- DXF directory: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123`

## Command
- `.venv-graph/bin/python scripts/train_2d_graph.py --manifest "reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_ODA_NORMALIZED_20260123.csv" --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123" --epochs 5 --batch-size 4 --hidden-dim 64 --lr 0.001 --node-dim 19 --downweight-label other --downweight-factor 0.3 --class-weighting none --sampler balanced --output "models/graph2d_training_oda_node19_normalized_balanced_20260123.pth"`

## Output
- `models/graph2d_training_oda_node19_normalized_balanced_20260123.pth`

## Notes
- Applied a balanced sampler to oversample minority labels in the training split.
