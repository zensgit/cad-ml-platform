# DEV_MECH_KNOWLEDGE_4000CAD_GRAPH2D_TEMPLATE_FEATURES_20260122

## Summary
- Added template-oriented node features (border/title-block hints) to the DXF graph pipeline.
- Trained a new Graph2D checkpoint with the enhanced features on the balanced manifest.
- Evaluated the template-feature model; template accuracy did not improve in this run.

## Code Changes
- `src/ml/train/dataset_2d.py` (node dim 9 with border/title-block hints)
- `scripts/train_2d_graph.py` (node-dim parameter)
- `scripts/eval_2d_graph.py` (node-dim aware dataset)
- `src/ml/vision_2d.py` (node-dim aware inference)

## Outputs
- Checkpoint: `models/graph2d_template_20260122.pth`
- Metrics: `reports/experiments/20260122/MECH_4000_DWG_GRAPH2D_VAL_METRICS_TEMPLATE_20260122.csv`
- Errors: `reports/experiments/20260122/MECH_4000_DWG_GRAPH2D_VAL_ERRORS_TEMPLATE_20260122.csv`

## Key Stats
- Validation summary: acc=0.414, top2=0.707 (58 samples)
- Template accuracy: 0.00 (0/4), top2=0.50
- 装配图 accuracy: 0.60 (3/5)

## Commands
- `./.venv-graph/bin/python scripts/train_2d_graph.py --manifest reports/experiments/20260122/MECH_4000_DWG_LABEL_MANIFEST_BALANCED_20260122.csv --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf --epochs 6 --downweight-label 机械制图 --downweight-factor 0.25 --output models/graph2d_template_20260122.pth --node-dim 9`
- `./.venv-graph/bin/python scripts/eval_2d_graph.py --manifest reports/experiments/20260122/MECH_4000_DWG_LABEL_MANIFEST_BALANCED_20260122.csv --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf --checkpoint models/graph2d_template_20260122.pth --output-metrics reports/experiments/20260122/MECH_4000_DWG_GRAPH2D_VAL_METRICS_TEMPLATE_20260122.csv --output-errors reports/experiments/20260122/MECH_4000_DWG_GRAPH2D_VAL_ERRORS_TEMPLATE_20260122.csv`

## Notes
- Template detection likely needs OCR or explicit title-block geometry extraction to improve.
