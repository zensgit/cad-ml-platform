# DEV_MECH_KNOWLEDGE_4000CAD_EXPANSION_VALIDATION_20260120B

## Summary
Trained the graph2d classifier on the expanded 4000CAD manifest (223 labeled rows)
with `练习零件图` downweighted, then validated the DXF fusion integration path.

## Training
- Command: `./.venv-graph/bin/python scripts/train_2d_graph.py --manifest reports/MECH_4000_DWG_LABEL_MANIFEST_20260119.csv --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf --downweight-label 练习零件图 --downweight-factor 0.3`
- Output:
  - `Epoch 1/3 loss=6.2882 val_acc=0.222`
  - `Epoch 2/3 loss=8.8642 val_acc=0.222`
  - `Epoch 3/3 loss=4.0814 val_acc=0.222`
- Checkpoint: `models/graph2d_latest.pth`

## Tests
- `GRAPH2D_ENABLED=true GRAPH2D_MODEL_PATH=models/graph2d_latest.pth pytest tests/integration/test_analyze_dxf_fusion.py -v`
- Result: `1 passed`
