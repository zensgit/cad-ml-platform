# DEV_TRAINING_DXF_SOFT_OVERRIDE_CALIBRATED_BATCH_ANALYSIS_20260124

## Objective
Re-run the DXF batch analysis with Graph2D temperature calibration enabled and compare soft-override candidate counts and confidence distributions against the pre-calibration run.

## Inputs
- Baseline batch run:
  - `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_soft_override_suggestion/`
- Calibrated batch run:
  - `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_soft_override_suggestion_calibrated_20260124/`
- Calibration JSON: `reports/experiments/20260123/GRAPH2D_TEMPERATURE_CALIBRATION_20260124.json`
- Model checkpoint: `models/graph2d_training_oda_node19_normalized_cleaned_augmented_20260123.pth`
- DXF directory: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123`

## Command
```
GRAPH2D_ENABLED=true GRAPH2D_FUSION_ENABLED=true FUSION_ANALYZER_ENABLED=true \
GRAPH2D_SOFT_OVERRIDE_MIN_CONF=0.17 \
GRAPH2D_MODEL_PATH=models/graph2d_training_oda_node19_normalized_cleaned_augmented_20260123.pth \
GRAPH2D_TEMPERATURE_CALIBRATION_PATH=reports/experiments/20260123/GRAPH2D_TEMPERATURE_CALIBRATION_20260124.json \
.venv-graph/bin/python scripts/batch_analyze_dxf_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123" \
  --output-dir "reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_soft_override_suggestion_calibrated_20260124" \
  --max-files 120
```

## Calibrated Run Summary
- Total: 110
- Success: 110
- Errors: 0
- Low-confidence count (<=0.6): 53
- Soft-override candidates: 27

## Implementation Updates
- `src/ml/vision_2d.py`: load Graph2D temperature from `GRAPH2D_TEMPERATURE` or `GRAPH2D_TEMPERATURE_CALIBRATION_PATH` and apply scaling to logits.
- `scripts/batch_analyze_dxf_local.py`: include `graph2d_temperature` and `graph2d_temperature_source` in batch CSV output.

## Comparison vs Baseline
### Soft-override candidates
- Baseline: 12
- Calibrated: 27
- Added candidates: 15
- Removed candidates: 0
- Added list: `reports/experiments/20260123/soft_override_calibrated_added_candidates_20260124.csv`

### Graph2D confidence distribution (all samples)
| Bucket | Baseline | Calibrated |
| --- | --- | --- |
| < 0.17 | 72 | 53 |
| 0.17–0.18 | 28 | 43 |
| 0.18–0.19 | 8 | 10 |
| 0.19–0.20 | 2 | 4 |

### Graph2D confidence distribution (soft-override candidates)
| Bucket | Baseline | Calibrated |
| --- | --- | --- |
| 0.17–0.18 | 8 | 23 |
| 0.18–0.19 | 2 | 2 |
| 0.19–0.20 | 2 | 2 |

### Average confidence
- Graph2D avg confidence: 0.169926 -> 0.171605
- Final classification avg confidence: 0.746364 (unchanged)

## Outputs
- Calibrated batch results: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_soft_override_suggestion_calibrated_20260124/batch_results.csv`
- Calibrated summary: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_soft_override_suggestion_calibrated_20260124/summary.json`
- Added candidates list: `reports/experiments/20260123/soft_override_calibrated_added_candidates_20260124.csv`

## Notes
- Temperature scaling increased Graph2D confidence slightly (T=0.965337), shifting 19 samples from <0.17 into >=0.17 buckets, which increased eligible soft-override candidates.
