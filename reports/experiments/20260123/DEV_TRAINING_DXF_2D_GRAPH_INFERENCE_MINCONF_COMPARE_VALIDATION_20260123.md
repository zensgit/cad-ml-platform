# DEV_TRAINING_DXF_2D_GRAPH_INFERENCE_MINCONF_COMPARE_VALIDATION_20260123

## Checks
- Executed two DXF batch analysis runs with `GRAPH2D_MIN_CONF` set to 0.5 and 0.6.
- Computed pass counts from Graph2D confidence values.

## Runtime Output
- Command (min_conf=0.5):
  - `GRAPH2D_MODEL_PATH=models/graph2d_training_cleaned_20260123.pth GRAPH2D_MIN_CONF=0.5 .venv-graph/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir "$DXF_DIR" --output-dir reports/experiments/20260123/dxf_batch_analysis_graph2d_cleaned_minconf05 --max-files 50 --seed 23 --min-confidence 0.6`
- Command (min_conf=0.6):
  - `GRAPH2D_MODEL_PATH=models/graph2d_training_cleaned_20260123.pth GRAPH2D_MIN_CONF=0.6 .venv-graph/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir "$DXF_DIR" --output-dir reports/experiments/20260123/dxf_batch_analysis_graph2d_cleaned_minconf06 --max-files 50 --seed 23 --min-confidence 0.6`
- Result:
  - min_conf=0.5: `passed=42/50`, avg_conf=0.7265
  - min_conf=0.6: `passed=39/50`, avg_conf=0.7265

## Notes
- Model hoster connectivity check message still appears even with `DISABLE_MODEL_SOURCE_CHECK=1`.
