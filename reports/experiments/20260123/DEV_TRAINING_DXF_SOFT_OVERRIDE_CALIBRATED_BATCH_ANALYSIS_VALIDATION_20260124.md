# DEV_TRAINING_DXF_SOFT_OVERRIDE_CALIBRATED_BATCH_ANALYSIS_VALIDATION_20260124

## Validation Summary
- Verified calibrated batch analysis outputs exist and summary matches expected counts.
- Confirmed candidate increase vs baseline (12 -> 27) and added-candidates list has 15 entries.
- Checked Graph2D confidence bucket distributions for baseline vs calibrated runs.

## Checks
- Calibrated summary file: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_soft_override_suggestion_calibrated_20260124/summary.json`
  - Total=110, Success=110, Errors=0, Low-confidence=53, Soft-override candidates=27
- Baseline summary file: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_soft_override_suggestion/summary.json`
  - Soft-override candidates=12
- Added candidates list: `reports/experiments/20260123/soft_override_calibrated_added_candidates_20260124.csv`
  - Rows=15
- Calibrated batch CSV includes temperature fields: `graph2d_temperature`, `graph2d_temperature_source`

## Distribution Checks (from batch_results.csv)
- Baseline Graph2D confidence buckets: <0.17=72, 0.17–0.18=28, 0.18–0.19=8, 0.19–0.20=2
- Calibrated Graph2D confidence buckets: <0.17=53, 0.17–0.18=43, 0.18–0.19=10, 0.19–0.20=4
- Baseline candidate buckets: 0.17–0.18=8, 0.18–0.19=2, 0.19–0.20=2
- Calibrated candidate buckets: 0.17–0.18=23, 0.18–0.19=2, 0.19–0.20=2
