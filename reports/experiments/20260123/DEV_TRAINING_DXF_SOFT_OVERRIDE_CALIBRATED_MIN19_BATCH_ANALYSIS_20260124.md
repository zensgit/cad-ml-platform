# DEV_TRAINING_DXF_SOFT_OVERRIDE_CALIBRATED_MIN19_BATCH_ANALYSIS_20260124

## Objective
Re-run calibrated batch analysis with `GRAPH2D_SOFT_OVERRIDE_MIN_CONF=0.19` after manual review flagged 0% precision in the 0.17–0.18 band.

## Inputs
- Review decision: `reports/experiments/20260123/soft_override_calibrated_added_review_decision_20260124.md`
- Calibration JSON: `reports/experiments/20260123/GRAPH2D_TEMPERATURE_CALIBRATION_20260124.json`
- Model checkpoint: `models/graph2d_training_oda_node19_normalized_cleaned_augmented_20260123.pth`
- DXF directory: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123`

## Command
```
GRAPH2D_ENABLED=true GRAPH2D_FUSION_ENABLED=true FUSION_ANALYZER_ENABLED=true \
GRAPH2D_SOFT_OVERRIDE_MIN_CONF=0.19 \
GRAPH2D_MODEL_PATH=models/graph2d_training_oda_node19_normalized_cleaned_augmented_20260123.pth \
GRAPH2D_TEMPERATURE_CALIBRATION_PATH=reports/experiments/20260123/GRAPH2D_TEMPERATURE_CALIBRATION_20260124.json \
.venv-graph/bin/python scripts/batch_analyze_dxf_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123" \
  --output-dir "reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_soft_override_suggestion_calibrated_min19_20260124" \
  --max-files 120
```

## Run Summary
- Total: 110
- Success: 110
- Errors: 0
- Low-confidence count (<=0.6): 53
- Soft-override candidates: 2

## Comparison vs Calibrated Min 0.17
- Previous candidates: 27
- Min 0.19 candidates: 2
- Removed candidates: 25
- Added candidates: 0
- Removed list: `reports/experiments/20260123/soft_override_calibrated_min19_removed_candidates_20260124.csv`

## Candidate Outputs
- Candidate list: `reports/experiments/20260123/soft_override_calibrated_min19_candidates_20260124.csv`
- Candidate label counts: `reports/experiments/20260123/soft_override_calibrated_min19_label_counts_20260124.csv`
- Candidate confidence buckets: `reports/experiments/20260123/soft_override_calibrated_min19_confidence_buckets_20260124.csv`

## Notes
- Remaining candidates are still labeled `传动件`; consider manual review before enabling overrides at 0.19.
