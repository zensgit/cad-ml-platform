# DEV_MECH_KNOWLEDGE_4000CAD_GRAPH2D_EVAL_20260121

## Summary
- Re-trained the Graph2D classifier for 6 epochs with class downweighting on 机械制图.
- Generated per-class validation metrics and an error bucket report.

## Inputs
- Manifest: `reports/experiments/20260121/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260121.csv`
- DXF source: `/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf`

## Outputs
- Checkpoint: `models/graph2d_merged_latest.pth`
- Validation metrics: `reports/experiments/20260121/MECH_4000_DWG_GRAPH2D_VAL_METRICS_20260121.csv`
- Validation errors: `reports/experiments/20260121/MECH_4000_DWG_GRAPH2D_VAL_ERRORS_20260121.csv`

## Key Stats
- Training epochs: 6 (downweight 机械制图=0.25)
- Final epoch: loss=1.8548, val_acc=0.400
- Validation summary: acc=0.400, top2=0.711, errors=27 (45 samples)
- Per-class highlights: 机械制图 acc=1.000 (16/16), 零件图 acc=0.000 (0/13)

## Commands
- `./.venv-graph/bin/python scripts/train_2d_graph.py --manifest reports/experiments/20260121/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260121.csv --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf --epochs 6 --downweight-label 机械制图 --downweight-factor 0.25 --output models/graph2d_merged_latest.pth`
- `./.venv-graph/bin/python scripts/eval_2d_graph.py --manifest reports/experiments/20260121/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260121.csv --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf --checkpoint models/graph2d_merged_latest.pth`

## Notes
- Validation split is randomized with seed=42 and val_split=0.2.
