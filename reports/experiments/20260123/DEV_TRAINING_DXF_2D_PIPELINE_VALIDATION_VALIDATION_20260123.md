# DEV_TRAINING_DXF_2D_PIPELINE_VALIDATION_VALIDATION_20260123

## Checks
- Ran the local batch DXF analyzer via FastAPI TestClient.
- Collected summary stats and low-confidence samples.

## Runtime Output
- Command (DXF_DIR set to a local training DXF directory with non-ASCII path):
  - `python3 scripts/batch_analyze_dxf_local.py --dxf-dir "$DXF_DIR" --output-dir reports/experiments/20260123/dxf_batch_analysis --max-files 50 --seed 23 --min-confidence 0.6`
- Result:
  - `total=50`, `success=50`, `error=0`
  - `confidence_buckets`: `0_4_0_6=14`, `0_6_0_8=13`, `gte_0_8=23`
  - `low_confidence_count=27`

## Notes
- Graph2D model was not loaded because torch was unavailable; graph2d fields in the CSV are empty.
- Label distribution details (including non-ASCII labels) are in `label_distribution.csv`.
