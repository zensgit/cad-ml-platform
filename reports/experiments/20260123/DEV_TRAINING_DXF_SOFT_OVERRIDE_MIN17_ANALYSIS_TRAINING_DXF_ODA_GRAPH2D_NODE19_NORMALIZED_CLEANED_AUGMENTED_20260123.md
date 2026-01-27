# DEV_TRAINING_DXF_SOFT_OVERRIDE_MIN17_ANALYSIS_TRAINING_DXF_ODA_GRAPH2D_NODE19_NORMALIZED_CLEANED_AUGMENTED_20260123

## Summary
- Simulated a soft override for L1 rule outputs (rule_version=v1, confidence_source=rules) when Graph2D confidence ≥ 0.17.

## Inputs
- Batch results: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented/batch_results.csv`
- Criteria:
  - rule_version == `v1`
  - confidence_source == `rules`
  - graph2d_confidence >= `0.17`

## Outputs
- Analysis folder: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented/soft_override_analysis_min17`
  - `soft_override_candidates.csv`
  - `soft_override_summary.json`
  - `soft_override_label_counts.csv` (post-override counts)
  - `soft_override_label_deltas.csv` (baseline vs post-override)

## Key Findings
- Candidates matching override criteria: 12.
- Override source labels: complex_assembly (6), moderate_component (6).
- Override target labels: 传动件 (6), 罐体 (4), 设备 (2).
- Net label deltas: complex_assembly -6, moderate_component -6, 传动件 +6, 罐体 +4, 设备 +2.

## Candidate List (file-level)
- See `soft_override_candidates.csv` for the 12 file-level entries and their graph2d_confidence values.
