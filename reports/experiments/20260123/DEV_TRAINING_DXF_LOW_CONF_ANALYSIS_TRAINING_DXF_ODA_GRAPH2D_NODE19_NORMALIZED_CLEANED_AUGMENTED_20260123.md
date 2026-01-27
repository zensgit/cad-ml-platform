# DEV_TRAINING_DXF_LOW_CONF_ANALYSIS_TRAINING_DXF_ODA_GRAPH2D_NODE19_NORMALIZED_CLEANED_AUGMENTED_20260123

## Summary
- Analyzed low-confidence predictions from the augmented Graph2D batch run to identify label patterns, confidence ranges, and fusion vs graph disagreement.

## Inputs
- Batch results: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented/batch_results.csv`
- Low-confidence subset: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented/batch_low_confidence.csv`
- Output analysis folder: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented/low_confidence_analysis`

## Command
- `python3 - <<'PY' ...` (ad-hoc pandas analysis; outputs saved to the analysis folder)

## Outputs
- Stats: `.../low_confidence_stats.json`
- Label distribution (low vs overall): `.../low_confidence_label_distribution.csv`
- Graph2D label distribution: `.../low_confidence_graph2d_label_distribution.csv`
- Fusion vs Graph2D crosstab: `.../low_confidence_fusion_vs_graph2d.csv`
- Confidence buckets: `.../low_confidence_buckets.csv`
- Confidence source counts: `.../low_confidence_source_counts.csv`
- Label mismatch counts: `.../low_confidence_label_mismatch.csv`
- Lowest-20 samples: `.../low_confidence_lowest_20.csv`

## Key Findings
- Low-confidence volume: 53 / 110 (48.18%).
- Confidence range: 0.55–0.60 (mean=0.5557, median=0.55).
- All low-confidence predictions use `confidence_source=rules`.
- Fusion label is `Standard_Part` for all low-confidence samples.
- Graph2D labels within low-confidence set: 传动件 (33), 罐体 (18), 设备 (2).
- Fusion vs Graph2D labels disagree for all 53 low-confidence entries.

## Notes
- The low-confidence cohort appears dominated by rule-driven outputs rather than Graph2D confidence.
- Next steps should focus on why rules override Graph2D outputs for these files and whether the rule thresholds are too aggressive.
