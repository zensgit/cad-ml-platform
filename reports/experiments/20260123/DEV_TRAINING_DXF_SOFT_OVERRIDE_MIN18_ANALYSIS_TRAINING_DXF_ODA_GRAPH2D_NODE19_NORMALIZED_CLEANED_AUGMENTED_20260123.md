# DEV_TRAINING_DXF_SOFT_OVERRIDE_MIN18_ANALYSIS_TRAINING_DXF_ODA_GRAPH2D_NODE19_NORMALIZED_CLEANED_AUGMENTED_20260123

## Summary
- Simulated a soft override for L1 rule outputs (rule_version=v1, confidence_source=rules) when Graph2D confidence ≥ 0.18.

## Inputs
- Batch results: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented/batch_results.csv`
- Criteria:
  - rule_version == `v1`
  - confidence_source == `rules`
  - graph2d_confidence >= `0.18`

## Outputs
- Analysis folder: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented/soft_override_analysis_min18`
  - `soft_override_candidates.csv`
  - `soft_override_summary.json`
  - `soft_override_label_counts.csv` (post-override counts)
  - `soft_override_label_deltas.csv` (baseline vs post-override)
  - `baseline_label_counts.csv`

## Key Findings
- Candidates matching override criteria: 4.
- Override source labels: complex_assembly (2), moderate_component (2).
- Override target labels: 传动件 (4).
- Net label deltas: complex_assembly -2, moderate_component -2, 传动件 +4; all other labels unchanged.

## Notes
- This is a simulated re-label; no model retraining or API changes were applied.
