# DEV_MECH_KNOWLEDGE_4000CAD_TEMPLATE_UPSAMPLE_20260122

## Summary
- Upsampled template-class entries to strengthen template detection.
- Trained and evaluated a new Graph2D checkpoint using the template-upsampled manifest.
- Template accuracy improved significantly in this run.

## Inputs
- Base manifest: `reports/experiments/20260122/MECH_4000_DWG_LABEL_MANIFEST_BALANCED_20260122.csv`
- DXF source: `/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf`

## Outputs
- Template-upsampled manifest: `reports/experiments/20260122/MECH_4000_DWG_LABEL_MANIFEST_TEMPLATE_UPSAMPLED_20260122.csv`
- Template-upsampled counts: `reports/experiments/20260122/MECH_4000_DWG_LABEL_COUNTS_TEMPLATE_UPSAMPLED_20260122.csv`
- Candidate list (manual review): `reports/experiments/20260122/MECH_4000_DWG_TEMPLATE_CANDIDATES_20260122.csv`
- Checkpoint: `models/graph2d_template_upsampled_20260122.pth`
- Metrics: `reports/experiments/20260122/MECH_4000_DWG_GRAPH2D_VAL_METRICS_TEMPLATE_UPSAMPLED_20260122.csv`
- Errors: `reports/experiments/20260122/MECH_4000_DWG_GRAPH2D_VAL_ERRORS_TEMPLATE_UPSAMPLED_20260122.csv`

## Key Stats
- Template count target: 80 (manifest rows: 337)
- Validation summary: acc=0.485, top2=0.691 (68 samples)
- Template accuracy: 0.80 (16/20), top2=1.00
- 装配图 accuracy: 0.33 (2/6)
- 零件图 accuracy: 0.00 (0/12)

## Commands
- Template manifest generation: (python inline)
- `./.venv-graph/bin/python scripts/train_2d_graph.py --manifest reports/experiments/20260122/MECH_4000_DWG_LABEL_MANIFEST_TEMPLATE_UPSAMPLED_20260122.csv --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf --epochs 6 --downweight-label 机械制图 --downweight-factor 0.25 --output models/graph2d_template_upsampled_20260122.pth --node-dim 9`
- `./.venv-graph/bin/python scripts/eval_2d_graph.py --manifest reports/experiments/20260122/MECH_4000_DWG_LABEL_MANIFEST_TEMPLATE_UPSAMPLED_20260122.csv --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf --checkpoint models/graph2d_template_upsampled_20260122.pth --output-metrics reports/experiments/20260122/MECH_4000_DWG_GRAPH2D_VAL_METRICS_TEMPLATE_UPSAMPLED_20260122.csv --output-errors reports/experiments/20260122/MECH_4000_DWG_GRAPH2D_VAL_ERRORS_TEMPLATE_UPSAMPLED_20260122.csv`

## Notes
- Candidate list is provided for manual confirmation before re-labeling additional template samples.
