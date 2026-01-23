# DEV_TRAINING_DXF_2D_GRAPH_TRAINING_VALIDATION_20260123

## Checks
- Trained the 2D DXF graph model using the training manifest and DXF directory.

## Runtime Output
- Command:
  - `.venv-graph/bin/python scripts/train_2d_graph.py --manifest reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_20260123.csv --dxf-dir "$DXF_DIR" --epochs 3 --batch-size 4 --output models/graph2d_training_20260123.pth`
- Result:
  - `Epoch 1/3 loss=4.0554 val_acc=0.000`
  - `Epoch 2/3 loss=3.6807 val_acc=0.000`
  - `Epoch 3/3 loss=3.4525 val_acc=0.000`
  - `Saved checkpoint: models/graph2d_training_20260123.pth`

## Notes
- Validation accuracy is low; consider label cleanup or class balancing before extended training.
