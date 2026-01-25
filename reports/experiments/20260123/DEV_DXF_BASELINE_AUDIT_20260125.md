# DEV_DXF_BASELINE_AUDIT_20260125

## Objective
Establish a baseline for DXF classification coverage, label distribution, and Graph2D confidence before advancing to later stages.

## Inputs
- Label manifest: `reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_ODA_NORMALIZED_CLEANED_20260123.csv`
- Graph2D batch results: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_soft_override_suggestion_calibrated_20260124/batch_results.csv`
- Synonyms: `data/knowledge/label_synonyms_template.json`

## Outputs
- Label distribution: `reports/experiments/20260123/baseline_label_distribution_20260125.csv`
- Filename coverage: `reports/experiments/20260123/baseline_filename_coverage_20260125.csv`
- Graph2D confidence buckets: `reports/experiments/20260123/baseline_graph2d_confidence_buckets_20260125.csv`
- Golden set template: `reports/experiments/20260123/golden_set_template_20260125.csv`

## Baseline Findings
### Label Coverage
- Synonym labels (global): 144
- Manifest labels (current sample): 8
- Total samples in manifest: 110

### Filename Coverage (manifest)
- Matched: 109 / 110 (99.1%)
- No match: 1 / 110 (0.9%)
- Extraction failed: 0

### Graph2D Confidence Distribution (batch)
- 0.10–0.17: 53
- 0.17–0.18: 43
- 0.18–0.19: 10
- 0.19–0.20: 4
- >= 0.20: 0

## Acceptance Targets (Recommended)
- Filename extraction coverage ≥ 80%
- Filename match rate ≥ 70%
- Fusion Top-1 accuracy ≥ 85%
- High-confidence (≥0.8) precision ≥ 95%
- Golden set size: 20–50 manually validated files

## Notes
- Current Graph2D confidence is heavily concentrated below 0.20, confirming weak geometry-only performance.
- Filename classifier appears high-coverage for this dataset; we should preserve a golden set that includes low/empty filename cases.
