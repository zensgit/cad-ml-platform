# DEV_MECH_KNOWLEDGE_4000CAD_GRAPH2D_TARGETED_AUGMENTATION_VALIDATION_20260122

## Checks
- Balanced manifest rows: 287
- Balanced label counts exported
- Balanced checkpoint saved
- Validation metrics + error buckets exported
- DXF fusion integration test passed with balanced model

## Tests
- `GRAPH2D_ENABLED=true GRAPH2D_MODEL_PATH=models/graph2d_balanced_20260122.pth FUSION_ANALYZER_ENABLED=true GRAPH2D_FUSION_ENABLED=true pytest tests/integration/test_analyze_dxf_fusion.py -v`

## Files Verified
- `reports/experiments/20260122/MECH_4000_DWG_LABEL_MANIFEST_BALANCED_20260122.csv`
- `reports/experiments/20260122/MECH_4000_DWG_LABEL_COUNTS_BALANCED_20260122.csv`
- `models/graph2d_balanced_20260122.pth`
- `reports/experiments/20260122/MECH_4000_DWG_GRAPH2D_VAL_METRICS_BALANCED_20260122.csv`
- `reports/experiments/20260122/MECH_4000_DWG_GRAPH2D_VAL_ERRORS_BALANCED_20260122.csv`
