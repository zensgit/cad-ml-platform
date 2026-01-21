# DEV_MECH_KNOWLEDGE_4000CAD_GRAPH2D_TRAINING_20260121

## Summary
- Trained Graph2D on the merged 4000CAD manifest with downweighted "机械制图" to balance class dominance.
- Ran DXF fusion integration test using the refreshed checkpoint.

## Training Command
- `./.venv-graph/bin/python scripts/train_2d_graph.py --manifest reports/experiments/20260121/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260121.csv --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf --epochs 3 --batch-size 4 --downweight-label 机械制图 --downweight-factor 0.3 --output models/graph2d_merged_latest.pth`

## Training Output
- Downweighting label '机械制图' (idx=1) with factor 0.30
- Epoch 1/3 loss=7.2481 val_acc=0.378
- Epoch 2/3 loss=5.8640 val_acc=0.356
- Epoch 3/3 loss=13.0482 val_acc=0.378
- Saved checkpoint: `models/graph2d_merged_latest.pth`

## Validation Test
- Command: `GRAPH2D_ENABLED=true GRAPH2D_MODEL_PATH=models/graph2d_merged_latest.pth ./.venv-graph/bin/python -m pytest tests/integration/test_analyze_dxf_fusion.py -v`
- Result: 1 passed (7 ezdxf DeprecationWarning warnings)
