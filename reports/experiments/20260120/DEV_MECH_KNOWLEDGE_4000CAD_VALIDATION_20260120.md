# DEV_MECH_KNOWLEDGE_4000CAD_VALIDATION_20260120

## Summary
Trained the graph2d classifier on the updated 4000CAD manifest and exercised the
DXF fusion path through the integration test with GRAPH2D enabled.

## Training
- Command: `./.venv-graph/bin/python scripts/train_2d_graph.py --manifest reports/MECH_4000_DWG_LABEL_MANIFEST_20260119.csv --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf`
- Output:
  - `Epoch 1/3 loss=4.8643 val_acc=0.000`
  - `Epoch 2/3 loss=3.6183 val_acc=0.100`
  - `Epoch 3/3 loss=3.5117 val_acc=0.100`
- Checkpoint: `models/graph2d_latest.pth`

## Tests
- `GRAPH2D_ENABLED=true GRAPH2D_MODEL_PATH=models/graph2d_latest.pth pytest tests/integration/test_analyze_dxf_fusion.py -v`
- Result: `1 passed`

## Notes
- Validation accuracy remains low due to the small, newly expanded label set; this run
  is intended as a pipeline smoke test.
