# DEV_DXF_TITLEBLOCK_OVERRIDE_FULL_ANALYSIS_20260125

## Objective
Quantify the impact of enabling title-block overrides on the full tuned batch when manual review is unavailable.

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
  --output-dir "reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_override_full_20260125" \
  --max-files 120
```

## Outputs
- `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_override_full_20260125/summary.json`
- `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_override_full_20260125/batch_results.csv`
- `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_override_full_20260125/label_distribution.csv`
- `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_override_full_20260125/batch_low_confidence.csv`
- `reports/experiments/20260123/titleblock_override_full_source_summary_20260125.csv`
- `reports/experiments/20260123/titleblock_override_full_diff_20260125.csv`
- `reports/experiments/20260123/titleblock_override_full_summary_20260125.csv`

## Findings
- Total analyzed: 110 (success 110).
- Hybrid sources: filename 97.27%, fusion 1.82%, titleblock 0.91%.
- Conflicts recorded: 2/110 (1.82%).
- Override changed label/source for 1 file:
  - `J1424042-00出料正压隔离器v2-yuantus.dxf` switched from `fusion` to `titleblock` with the same label.

## Notes
- Override did not change the final label on conflicted samples; only one source attribution changed.
