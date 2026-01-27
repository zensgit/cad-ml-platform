# DEV_GRAPH2D_ENSEMBLE_INTEGRATION_VALIDATION_20260127

## Validation Summary
- Ensemble flag enabled: `GRAPH2D_ENSEMBLE_ENABLED=true`
- Smoke batch size: 5 DXF files
- Result: batch completed successfully and ensemble metadata is present in the CSV output

## Evidence
### Smoke Batch Artifacts
- `reports/experiments/20260127/dxf_batch_analysis_ensemble_smoke_20260127_r2/batch_results.csv`
- `reports/experiments/20260127/dxf_batch_analysis_ensemble_smoke_20260127_r2/summary.json`

### Key Observations
- Ensemble models loaded:
  - `models/graph2d_edge_sage_v3.pth`
  - `models/graph2d_edge_sage_v4_best.pth`
- Ensemble metadata columns are populated:
  - `graph2d_ensemble_enabled=True`
  - `graph2d_ensemble_size=2`
  - `graph2d_voting=soft`

## Command Used
```bash
GRAPH2D_ENABLED=true \
GRAPH2D_ENSEMBLE_ENABLED=true \
TITLEBLOCK_OVERRIDE_ENABLED=false \
DISABLE_MODEL_SOURCE_CHECK=1 \
  .venv-graph/bin/python scripts/batch_analyze_dxf_local.py \
    --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
    --max-files 5 \
    --seed 20260127 \
    --output-dir "reports/experiments/20260127/dxf_batch_analysis_ensemble_smoke_20260127_r2"
```

## Caveats
- Unit test suite could not be fully executed in this environment due to missing dependency `jwt` during collection.
