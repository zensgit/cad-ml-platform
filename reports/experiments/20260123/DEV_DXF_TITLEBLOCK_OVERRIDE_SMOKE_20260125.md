# DEV_DXF_TITLEBLOCK_OVERRIDE_SMOKE_20260125

## Objective
Run a small grey (override-enabled) batch to confirm title-block overrides do not supersede high-confidence filename labels.

## Execution
- Command:
```
TITLEBLOCK_ENABLED=true TITLEBLOCK_OVERRIDE_ENABLED=true TITLEBLOCK_MIN_CONF=0.75 \
TITLEBLOCK_FUSION_WEIGHT=0.2 TITLEBLOCK_REGION_X_RATIO=0.7 TITLEBLOCK_REGION_Y_RATIO=0.5 \
GRAPH2D_ENABLED=true GRAPH2D_FUSION_ENABLED=true FUSION_ANALYZER_ENABLED=true \
GRAPH2D_SOFT_OVERRIDE_MIN_CONF=0.19 \
GRAPH2D_MODEL_PATH=models/graph2d_training_oda_node19_normalized_cleaned_augmented_20260123.pth \
GRAPH2D_TEMPERATURE_CALIBRATION_PATH=reports/experiments/20260123/GRAPH2D_TEMPERATURE_CALIBRATION_20260124.json \
.venv-graph/bin/python scripts/batch_analyze_dxf_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123" \
  --output-dir "reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_override_smoke_20260125" \
  --max-files 30
```

## Outputs
- `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_override_smoke_20260125/summary.json`
- `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_override_smoke_20260125/batch_results.csv`
- `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_override_smoke_20260125/label_distribution.csv`
- `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_override_smoke_20260125/batch_low_confidence.csv`
- `reports/experiments/20260123/titleblock_override_smoke_summary_20260125.csv`

## Findings
- Total analyzed: 30 (success 30).
- Title-block overrides adopted: 0/30 (0%).
- Hybrid conflicts recorded: see `titleblock_override_smoke_summary_20260125.csv`.

## Notes
- With high-confidence filename labels present, override path remains inactive as intended.
