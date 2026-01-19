# DEV_MECH_KNOWLEDGE_2D_GRAPH_TRAINING_20260119

## Summary
Trained the 2D graph classifier on the converted DXF dataset and verified
inference on a sample DXF file.

## Training Command
- `./.venv-graph/bin/python scripts/train_2d_graph.py --manifest reports/MECH_DWG_LABEL_MANIFEST_20260119.csv --dxf-dir /Users/huazhou/Downloads/训练图纸/训练图纸_dxf`

## Training Results
- Epoch 1/3 loss=4.1101 val_acc=0.045
- Epoch 2/3 loss=3.6368 val_acc=0.000
- Epoch 3/3 loss=3.5033 val_acc=0.000
- Checkpoint: `models/graph2d_latest.pth`

## Inference Dry-Run
- Sample: `BTJ01230901522-00汽水分离器v1.dxf`
- Result: `{'label': '上封头组件', 'confidence': 0.0288, 'status': 'ok'}`

## Notes
- The dataset is weakly labeled from filenames; accuracy is expected to improve after label review.
