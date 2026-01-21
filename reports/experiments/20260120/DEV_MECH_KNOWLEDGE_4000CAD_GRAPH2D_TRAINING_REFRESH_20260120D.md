# DEV_MECH_KNOWLEDGE_4000CAD_GRAPH2D_TRAINING_REFRESH_20260120D

## Summary
Retrained the Graph2D checkpoint after clearing all review conflicts and
finalizing the merged manifest.

## Command
```
./.venv-graph/bin/python scripts/train_2d_graph.py \
  --manifest reports/experiments/20260120/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260120.csv \
  --dxf-dir "/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf" \
  --output models/graph2d_merged_latest.pth
```

## Result
- Epoch 1/3 loss=3.5704 val_acc=0.400
- Epoch 2/3 loss=12.0441 val_acc=0.467
- Epoch 3/3 loss=2.5302 val_acc=0.400
- Saved checkpoint: `models/graph2d_merged_latest.pth`
