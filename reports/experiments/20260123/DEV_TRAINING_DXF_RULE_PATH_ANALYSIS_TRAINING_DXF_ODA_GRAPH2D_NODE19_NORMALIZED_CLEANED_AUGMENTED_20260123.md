# DEV_TRAINING_DXF_RULE_PATH_ANALYSIS_TRAINING_DXF_ODA_GRAPH2D_NODE19_NORMALIZED_CLEANED_AUGMENTED_20260123

## Summary
- Traced rule vs fusion decision paths using the augmented batch results, and captured a sample low-confidence classification to inspect the rule-based fallback details.

## Inputs
- Batch results: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented/batch_results.csv`
- Low-confidence subset: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented/batch_low_confidence.csv`
- Source logic: `src/core/analyzer.py` (L1 rules), `src/api/v1/analyze.py` (fusion routing)

## Outputs
- Rule path analysis folder: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented/rule_path_analysis`
  - `rule_version_counts.csv`
  - `confidence_source_counts.csv`
  - `rule_version_confidence_stats.csv`
  - `low_confidence_by_rule_version.csv`
  - `low_confidence_part_type_counts.csv`
  - `part_type_counts.csv`
  - `graph2d_label_counts.csv`
  - `rule_path_summary.json`
  - `sample_rule_path_classification.json`

## Key Findings
- Rule version split: L2-Fusion-v1=57, v1=53.
- Confidence source split: fusion=57, rules=53.
- All low-confidence rows map to rule_version=v1 (L1 rules), part_type ∈ {complex_assembly, moderate_component}.
- Graph2D labels (all rows): 传动件=74, 罐体=28, 设备=8.
- Sample low-confidence classification shows FusionAnalyzer source=rule_based with RULE_DEFAULT fallback and confidence=0.55 (see `sample_rule_path_classification.json`).

## Notes
- The sampled run was executed with system Python (torch unavailable), so Graph2D inference was disabled for that sample; batch CSVs still reflect Graph2D outputs from the earlier venv-graph batch runs.
