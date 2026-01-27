# DEV_TRAINING_DXF_SOFT_OVERRIDE_SWEEP_ANALYSIS_TRAINING_DXF_ODA_GRAPH2D_NODE19_NORMALIZED_CLEANED_AUGMENTED_20260123

## Summary
- Swept soft-override thresholds (0.16/0.17/0.18/0.19) to quantify how many v1-rule outputs would be replaced by Graph2D labels.

## Inputs
- Batch results: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented/batch_results.csv`
- Criteria:
  - rule_version == `v1`
  - confidence_source == `rules`
  - graph2d_confidence >= threshold

## Outputs
- Sweep folder: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented/soft_override_sweep`
  - `soft_override_sweep_summary.csv`
  - `soft_override_candidates_min16.csv`
  - `soft_override_candidates_min17.csv`
  - `soft_override_candidates_min18.csv`
  - `soft_override_candidates_min19.csv`

## Key Findings
- Candidate counts by threshold:
  - 0.16 → 50
  - 0.17 → 12
  - 0.18 → 4
  - 0.19 → 2
- Override target labels by threshold:
  - 0.16 → 传动件 33, 罐体 15, 设备 2
  - 0.17 → 传动件 6, 罐体 4, 设备 2
  - 0.18 → 传动件 4
  - 0.19 → 传动件 2

## Notes
- Thresholds ≥0.18 only impact 传动件 predictions; at 0.16–0.17, 罐体/设备 become dominant overrides.
