# DEV_TRAINING_DXF_2D_GRAPH_TRAINING_CLEANED_VALIDATION_20260123

## Checks
- Trained the 2D DXF graph model using the cleaned manifest and downweighting for `other`.

## Runtime Output
- Command:
  - `.venv-graph/bin/python scripts/train_2d_graph.py --manifest reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_CLEANED_20260123.csv --dxf-dir "$DXF_DIR" --epochs 15 --batch-size 4 --output models/graph2d_training_cleaned_20260123.pth --downweight-label other --downweight-factor 0.3`
- Result:
  - `Epoch 1/15 loss=2.6272 val_acc=0.591`
  - `Epoch 12/15 loss=0.9918 val_acc=0.682`
  - `Epoch 15/15 loss=0.8431 val_acc=0.636`
  - `Saved checkpoint: models/graph2d_training_cleaned_20260123.pth`
