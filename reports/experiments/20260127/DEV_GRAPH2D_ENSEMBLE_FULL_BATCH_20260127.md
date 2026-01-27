# DEV_GRAPH2D_ENSEMBLE_FULL_BATCH_20260127

## Goal
Run the full DXF batch (requested 300, capped at 110 available) with Graph2D ensemble enabled in the `/api/v1/analyze` pipeline, and generate review artifacts.

## Command
```bash
GRAPH2D_ENABLED=true \
GRAPH2D_ENSEMBLE_ENABLED=true \
DISABLE_MODEL_SOURCE_CHECK=1 \
TITLEBLOCK_OVERRIDE_ENABLED=false \
.venv-graph/bin/python scripts/batch_analyze_dxf_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --max-files 300 \
  --seed 20260127 \
  --output-dir "reports/experiments/20260127/dxf_batch_analysis_training_dxf_random110_ensemble_full_20260127"
```

## Artifacts
All outputs were written under:
- `reports/experiments/20260127/dxf_batch_analysis_training_dxf_random110_ensemble_full_20260127`

Key files:
- `reports/experiments/20260127/dxf_batch_analysis_training_dxf_random110_ensemble_full_20260127/batch_results.csv`
- `reports/experiments/20260127/dxf_batch_analysis_training_dxf_random110_ensemble_full_20260127/summary.json`
- `reports/experiments/20260127/dxf_batch_analysis_training_dxf_random110_ensemble_full_20260127/soft_override_reviewed_20260127.csv`
- `reports/experiments/20260127/dxf_batch_analysis_training_dxf_random110_ensemble_full_20260127/ensemble_summary_20260127.csv`

## Headline Results
- Total samples: 110
- Success: 110, Errors: 0
- Ensemble enabled: 110/110 (100%)
- Drawing-type excluded: 56/110 (50.91%)
- Hybrid source: filename 107, fusion 3
- Filename coverage: 109/110 (99.09%)
- Auto review vs Graph2D: agree 0, disagree 109, unknown 1

## Notes
- The pipeline is working end-to-end with ensemble enabled and documented metadata.
- On this dataset, filename dominates the final decision (by design).
