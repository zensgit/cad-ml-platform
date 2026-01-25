# DEV_DXF_TITLEBLOCK_TUNED_FULL_BATCH_ANALYSIS_VALIDATION_20260125

## Validation Summary
- Verified tuned full-batch artifacts and coverage summaries after attribute-based title-block extraction.

## Artifacts
- `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_tuned_full_20260125/summary.json`
- `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_tuned_full_20260125/batch_results.csv`
- `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_tuned_full_20260125/label_distribution.csv`
- `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_tuned_full_20260125/batch_low_confidence.csv`
- `reports/experiments/20260123/titleblock_tuned_full_coverage_20260125.csv`
- `reports/experiments/20260123/titleblock_tuned_full_raw_texts_buckets_20260125.csv`
- `reports/experiments/20260123/titleblock_tuned_full_region_entities_buckets_20260125.csv`
- `reports/experiments/20260123/titleblock_tuned_full_raw_text_samples_20260125.csv`

## Checks
- `summary.json` reports total=110 and success=110.
- `titleblock_tuned_full_coverage_20260125.csv` shows 108/110 label matches and 110/110 samples with title-block text detected.
- `batch_results.csv` contains populated `titleblock_label` and `titleblock_confidence` values.
