# DEV_MECH_KNOWLEDGE_4000CAD_TRAINING_DOWNWEIGHT_20260120

## Summary
Downweighted the `练习零件图` class during graph2d training to reduce noise from
numeric-only practice drawings while keeping the samples in the dataset. Re-ran the
DXF fusion integration test with the updated checkpoint.

## Training
- Command: `./.venv-graph/bin/python scripts/train_2d_graph.py --manifest reports/MECH_4000_DWG_LABEL_MANIFEST_20260119.csv --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf --downweight-label 练习零件图 --downweight-factor 0.3`
- Output:
  - `Downweighting label '练习零件图' (idx=8) with factor 0.30`
  - `Epoch 1/3 loss=4.8643 val_acc=0.000`
  - `Epoch 2/3 loss=3.6183 val_acc=0.100`
  - `Epoch 3/3 loss=3.5117 val_acc=0.100`
- Checkpoint: `models/graph2d_latest.pth`

## Tests
- `GRAPH2D_ENABLED=true GRAPH2D_MODEL_PATH=models/graph2d_latest.pth pytest tests/integration/test_analyze_dxf_fusion.py -v`
- Result: `1 passed`

## Notes
- Manual review remains deferred; the downweighting keeps practice drawings but
  reduces their impact on the loss.
