# DEV_TRAINING_DXF_SOFT_OVERRIDE_ANALYSIS_TRAINING_DXF_ODA_GRAPH2D_NODE19_NORMALIZED_CLEANED_AUGMENTED_20260123

## Summary
- Simulated a “soft override” where L1 rule results (rule_version=v1, confidence_source=rules) would be replaced by Graph2D predictions if Graph2D confidence ≥ 0.60.

## Inputs
- Batch results: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented/batch_results.csv`
- Criteria:
  - rule_version == `v1`
  - confidence_source == `rules`
  - graph2d_confidence >= `0.60`

## Outputs
- Analysis folder: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented/soft_override_analysis`
  - `soft_override_candidates.csv`
  - `soft_override_summary.json`
  - `soft_override_label_counts.csv`
  - `soft_override_label_deltas.csv`
  - `baseline_label_counts.csv`

## Key Findings
- Candidates matching override criteria: 0.
- Graph2D confidence for v1-rule rows ranges ~0.158–0.194 (mean ~0.169), far below 0.60.
- With current scores, the soft override does not apply; label deltas are zero by construction.

## Notes
- If we want a soft override to activate, we either need to lower the threshold or improve Graph2D calibration/confidence.
