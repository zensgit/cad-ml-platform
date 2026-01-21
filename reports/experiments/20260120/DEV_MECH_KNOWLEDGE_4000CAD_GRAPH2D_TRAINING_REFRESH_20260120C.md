# DEV_MECH_KNOWLEDGE_4000CAD_GRAPH2D_TRAINING_REFRESH_20260120C

## Summary
Retrained the Graph2D checkpoint after applying label-merge rules in the merged
manifest.

## Command
```
./.venv-graph/bin/python scripts/train_2d_graph.py \
  --manifest reports/experiments/20260120/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260120.csv \
  --dxf-dir "/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf" \
  --output models/graph2d_merged_latest.pth
```

## Result
- Epoch 1/3 loss=10.9634 val_acc=0.400
- Epoch 2/3 loss=22.0067 val_acc=0.467
- Epoch 3/3 loss=11.7540 val_acc=0.467
- Saved checkpoint: `models/graph2d_merged_latest.pth`
