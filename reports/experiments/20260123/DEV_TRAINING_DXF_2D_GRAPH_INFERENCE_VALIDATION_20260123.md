# DEV_TRAINING_DXF_2D_GRAPH_INFERENCE_VALIDATION_20260123

## Checks
- Executed a DXF batch analysis run with `GRAPH2D_MODEL_PATH` pointing to the new checkpoint.
- Measured Graph2D confidence distribution.

## Runtime Output
- Command:
  - `GRAPH2D_MODEL_PATH=models/graph2d_training_20260123.pth .venv-graph/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir "$DXF_DIR" --output-dir reports/experiments/20260123/dxf_batch_analysis_graph2d_training --max-files 50 --seed 23 --min-confidence 0.6`
- Result:
  - `total=50`, `success=50`, `error=0`
  - Graph2D confidence buckets: `lt_0_4=46`, `0_4_0_6=0`, `0_6_0_8=0`, `gte_0_8=4`
  - Average Graph2D confidence: `0.1201`

## Notes
- Graph2D predictions are present in `batch_results.csv` but confidence is mostly low.
