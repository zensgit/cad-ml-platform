# DEV_DXF_TITLEBLOCK_BATCH_ANALYSIS_VALIDATION_20260125

## Validation Summary
- Verified batch analysis artifacts and title-block coverage summaries for the ODA training DXF run.

## Artifacts
- `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_20260125/summary.json`
- `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_20260125/batch_results.csv`
- `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_20260125/label_distribution.csv`
- `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_20260125/batch_low_confidence.csv`
- `reports/experiments/20260123/titleblock_coverage_20260125.csv`
- `reports/experiments/20260123/titleblock_raw_texts_buckets_20260125.csv`
- `reports/experiments/20260123/titleblock_region_entities_buckets_20260125.csv`

## Checks
- `summary.json` reports total=110 and success=110.
- `batch_results.csv` includes title-block columns: `titleblock_label`, `titleblock_confidence`, `titleblock_raw_texts_count`, `titleblock_region_entities_count`.
- `titleblock_coverage_20260125.csv` confirms 0 labeled samples and 40/110 samples with title-block raw text detected.
