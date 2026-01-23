# DEV_TRAINING_DXF_2D_GRAPH_FUSION_IMPACT_VALIDATION_20260123

## Checks
- Ran DXF batch analysis with FusionAnalyzer override disabled and enabled.
- Computed classification changes between the two runs.

## Runtime Output
- No override:
  - `GRAPH2D_ENABLED=true GRAPH2D_MODEL_PATH=models/graph2d_training_cleaned_20260123.pth GRAPH2D_MIN_CONF=0.6 GRAPH2D_FUSION_ENABLED=true FUSION_ANALYZER_ENABLED=true FUSION_ANALYZER_OVERRIDE=false FUSION_ANALYZER_OVERRIDE_MIN_CONF=0.6 .venv-graph/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir "$DXF_DIR" --output-dir reports/experiments/20260123/dxf_batch_analysis_graph2d_fusion_no_override --max-files 50 --seed 23 --min-confidence 0.6`
- Override enabled:
  - `GRAPH2D_ENABLED=true GRAPH2D_MODEL_PATH=models/graph2d_training_cleaned_20260123.pth GRAPH2D_MIN_CONF=0.6 GRAPH2D_FUSION_ENABLED=true FUSION_ANALYZER_ENABLED=true FUSION_ANALYZER_OVERRIDE=true FUSION_ANALYZER_OVERRIDE_MIN_CONF=0.6 .venv-graph/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir "$DXF_DIR" --output-dir reports/experiments/20260123/dxf_batch_analysis_graph2d_fusion_override --max-files 50 --seed 23 --min-confidence 0.6`

## Results
- Total samples: 50
- Classification changes: 35
- Changed to `other`: 33
- Changed to `再沸器`: 2
- Diff CSV: `reports/experiments/20260123/dxf_batch_analysis_graph2d_fusion_compare/override_diff.csv`

## Notes
- Model hoster connectivity check message still appears even with `DISABLE_MODEL_SOURCE_CHECK=1`.
