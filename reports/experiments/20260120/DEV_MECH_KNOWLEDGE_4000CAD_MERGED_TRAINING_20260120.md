# DEV_MECH_KNOWLEDGE_4000CAD_MERGED_TRAINING_20260120

## Summary
Trained a graph2d checkpoint on the merged-label manifest and validated the DXF
fusion integration path.

## Training
- Command: `./.venv-graph/bin/python scripts/train_2d_graph.py --manifest reports/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260120.csv --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf --downweight-label 练习零件图 --downweight-factor 0.3 --output models/graph2d_merged_latest.pth`
- Output:
  - `Epoch 1/3 loss=7.0581 val_acc=0.222`
  - `Epoch 2/3 loss=4.9884 val_acc=0.222`
  - `Epoch 3/3 loss=3.7277 val_acc=0.244`
- Checkpoint: `models/graph2d_merged_latest.pth`

## Tests
- `GRAPH2D_ENABLED=true GRAPH2D_MODEL_PATH=models/graph2d_merged_latest.pth pytest tests/integration/test_analyze_dxf_fusion.py -v`
- Result: `1 passed`
