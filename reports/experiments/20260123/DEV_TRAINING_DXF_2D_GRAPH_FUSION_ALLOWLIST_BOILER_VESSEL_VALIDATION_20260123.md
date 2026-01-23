# DEV_TRAINING_DXF_2D_GRAPH_FUSION_ALLOWLIST_BOILER_VESSEL_VALIDATION_20260123

## Checks
- Ran allowlist `再沸器,罐体部分` with `GRAPH2D_MIN_CONF=0.7` and override enabled.

## Runtime Output
- Command:
  - `GRAPH2D_ENABLED=true GRAPH2D_MODEL_PATH=models/graph2d_training_cleaned_20260123.pth GRAPH2D_MIN_CONF=0.7 GRAPH2D_ALLOW_LABELS=再沸器,罐体部分 GRAPH2D_EXCLUDE_LABELS=other GRAPH2D_FUSION_ENABLED=true FUSION_ANALYZER_ENABLED=true FUSION_ANALYZER_OVERRIDE=true FUSION_ANALYZER_OVERRIDE_MIN_CONF=0.6 .venv-graph/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir "$DXF_DIR" --output-dir reports/experiments/20260123/dxf_batch_analysis_graph2d_fusion_allowlist_boiler_vessel --max-files 50 --seed 23 --min-confidence 0.6`

## Results
- Total samples: 50
- Classification changes vs baseline: 2
- Diff CSV: `reports/experiments/20260123/dxf_batch_analysis_graph2d_fusion_allowlist_boiler_vessel/override_diff.csv`
