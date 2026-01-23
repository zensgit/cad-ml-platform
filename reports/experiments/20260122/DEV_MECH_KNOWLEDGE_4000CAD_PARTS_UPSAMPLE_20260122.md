# DEV_MECH_KNOWLEDGE_4000CAD_PARTS_UPSAMPLE_20260122

## Summary
- Upsampled the 零件图 class to address zero-accuracy regressions.
- Trained and evaluated a new Graph2D checkpoint with the parts-upsampled manifest.

## Outputs
- Parts-upsampled manifest: `reports/experiments/20260122/MECH_4000_DWG_LABEL_MANIFEST_PARTS_UPSAMPLED_20260122.csv`
- Parts-upsampled counts: `reports/experiments/20260122/MECH_4000_DWG_LABEL_COUNTS_PARTS_UPSAMPLED_20260122.csv`
- Checkpoint: `models/graph2d_parts_upsampled_20260122.pth`
- Metrics: `reports/experiments/20260122/MECH_4000_DWG_GRAPH2D_VAL_METRICS_PARTS_UPSAMPLED_20260122.csv`
- Errors: `reports/experiments/20260122/MECH_4000_DWG_GRAPH2D_VAL_ERRORS_PARTS_UPSAMPLED_20260122.csv`

## Key Stats
- Parts target: 100 (manifest rows: 367)
- Validation summary: acc=0.581, top2=0.797 (74 samples)
- 零件图 accuracy: 0.737 (14/19)
- 模板 accuracy: 1.000 (17/17)
- 机械制图 accuracy: 0.500 (10/20)

## Commands
- Parts manifest generation: (python inline)
- `./.venv-graph/bin/python scripts/train_2d_graph.py --manifest reports/experiments/20260122/MECH_4000_DWG_LABEL_MANIFEST_PARTS_UPSAMPLED_20260122.csv --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf --epochs 6 --downweight-label 机械制图 --downweight-factor 0.25 --output models/graph2d_parts_upsampled_20260122.pth --node-dim 9`
- `./.venv-graph/bin/python scripts/eval_2d_graph.py --manifest reports/experiments/20260122/MECH_4000_DWG_LABEL_MANIFEST_PARTS_UPSAMPLED_20260122.csv --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf --checkpoint models/graph2d_parts_upsampled_20260122.pth --output-metrics reports/experiments/20260122/MECH_4000_DWG_GRAPH2D_VAL_METRICS_PARTS_UPSAMPLED_20260122.csv --output-errors reports/experiments/20260122/MECH_4000_DWG_GRAPH2D_VAL_ERRORS_PARTS_UPSAMPLED_20260122.csv`
