# DEV_TRAINING_DXF_2D_GRAPH_INFERENCE_CLEANED_VALIDATION_20260123

## Checks
- Executed DXF batch analysis with the cleaned Graph2D checkpoint and confidence gate.
- Summarized Graph2D confidence buckets.

## Runtime Output
- Command:
  - `GRAPH2D_MODEL_PATH=models/graph2d_training_cleaned_20260123.pth GRAPH2D_MIN_CONF=0.4 .venv-graph/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir "$DXF_DIR" --output-dir reports/experiments/20260123/dxf_batch_analysis_graph2d_cleaned_training --max-files 50 --seed 23 --min-confidence 0.6`
- Result:
  - `gte_0_8=24`, `0_6_0_8=15`, `0_4_0_6=5`, `lt_0_4=6`
  - Average Graph2D confidence: `0.7265`
  - Passed threshold (>=0.4): `44/50`

## Notes
- Graph2D confidence improved vs the uncleaned model.
