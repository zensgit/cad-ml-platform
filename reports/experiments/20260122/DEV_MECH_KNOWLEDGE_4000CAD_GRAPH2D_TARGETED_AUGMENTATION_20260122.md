# DEV_MECH_KNOWLEDGE_4000CAD_GRAPH2D_TARGETED_AUGMENTATION_20260122

## Summary
- Identified top misclassified labels (零件图/装配图/模板) from the latest Graph2D error report.
- Created a balanced manifest by upsampling those labels and retrained Graph2D.
- Evaluated the balanced checkpoint and re-ran the DXF fusion integration tests with the new model.

## Inputs
- Base manifest: `reports/experiments/20260121/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260121.csv`
- Error review: `reports/experiments/20260121/MECH_4000_DWG_GRAPH2D_VAL_ERRORS_20260121.csv`
- DXF source: `/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf`

## Outputs
- Balanced manifest: `reports/experiments/20260122/MECH_4000_DWG_LABEL_MANIFEST_BALANCED_20260122.csv`
- Balanced label counts: `reports/experiments/20260122/MECH_4000_DWG_LABEL_COUNTS_BALANCED_20260122.csv`
- Balanced checkpoint: `models/graph2d_balanced_20260122.pth`
- Balanced metrics: `reports/experiments/20260122/MECH_4000_DWG_GRAPH2D_VAL_METRICS_BALANCED_20260122.csv`
- Balanced errors: `reports/experiments/20260122/MECH_4000_DWG_GRAPH2D_VAL_ERRORS_BALANCED_20260122.csv`

## Key Stats
- Target counts: 零件图=70, 装配图=40, 模板=30 (balanced manifest rows: 287)
- Validation summary: acc=0.431, top2=0.672 (58 samples)
- Per-class improvements vs previous:
  - 装配图 acc 0.40 → 0.60
  - 零件图 acc 0.00 → 0.08
  - 模板 acc remains 0.00 (needs more signal/features)

## Commands
- Balanced manifest generation: (python inline)
- `./.venv-graph/bin/python scripts/train_2d_graph.py --manifest reports/experiments/20260122/MECH_4000_DWG_LABEL_MANIFEST_BALANCED_20260122.csv --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf --epochs 6 --downweight-label 机械制图 --downweight-factor 0.25 --output models/graph2d_balanced_20260122.pth`
- `./.venv-graph/bin/python scripts/eval_2d_graph.py --manifest reports/experiments/20260122/MECH_4000_DWG_LABEL_MANIFEST_BALANCED_20260122.csv --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf --checkpoint models/graph2d_balanced_20260122.pth --output-metrics reports/experiments/20260122/MECH_4000_DWG_GRAPH2D_VAL_METRICS_BALANCED_20260122.csv --output-errors reports/experiments/20260122/MECH_4000_DWG_GRAPH2D_VAL_ERRORS_BALANCED_20260122.csv`
- `GRAPH2D_ENABLED=true GRAPH2D_MODEL_PATH=models/graph2d_balanced_20260122.pth FUSION_ANALYZER_ENABLED=true GRAPH2D_FUSION_ENABLED=true pytest tests/integration/test_analyze_dxf_fusion.py -v`

## Notes
- Balanced manifest is created via deterministic row duplication; no new DXF files were added.
