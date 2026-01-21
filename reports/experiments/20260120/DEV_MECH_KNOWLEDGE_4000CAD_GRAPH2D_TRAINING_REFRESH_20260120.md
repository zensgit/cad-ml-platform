# DEV_MECH_KNOWLEDGE_4000CAD_GRAPH2D_TRAINING_REFRESH_20260120

## Summary
Refreshed the merged-label Graph2D checkpoint using the 4000CAD DXF conversion
output to keep the default model aligned with the merged manifest.

## Dataset
- Manifest: `reports/experiments/20260120/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260120.csv`
- DXF dir: `/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf`

## Training
Command:
```
./.venv-graph/bin/python scripts/train_2d_graph.py \
  --manifest reports/experiments/20260120/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260120.csv \
  --dxf-dir "/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf" \
  --output models/graph2d_merged_latest.pth
```

Result:
- Epoch 1/3 loss=7.0581 val_acc=0.222
- Epoch 2/3 loss=4.9884 val_acc=0.222
- Epoch 3/3 loss=3.7277 val_acc=0.244
- Saved: `models/graph2d_merged_latest.pth`

## Config
Updated the training default DXF directory in `scripts/train_2d_graph.py` to
match the 4000CAD DXF output location.
