# DEV_VISION_CAD_FEATURE_LARGE_DATASET_COMPARE_VALIDATION_20260106

## Scope
Run the CAD feature benchmark on a larger real dataset and validate comparison
output using the generated baseline.

## Commands
- `python3 scripts/vision_cad_feature_benchmark.py --no-clients --input-dir data/dedup_report_train_local_version_profile_spatial_full_package/assets/images --max-samples 20 --output-json reports/vision_cad_feature_baseline_spatial_20260106.json`
- `python3 scripts/vision_cad_feature_benchmark.py --no-clients --input-dir data/dedup_report_train_local_version_profile_spatial_full_package/assets/images --max-samples 20 --output-json reports/vision_cad_feature_baseline_spatial_compare_20260106.json --compare-json reports/vision_cad_feature_baseline_spatial_20260106.json`

## Results
- `total_samples=20`, `total_combos=1`
- `comparison.summary_delta` values are all zero for the baseline run

## Outputs
- `reports/vision_cad_feature_baseline_spatial_20260106.json`
- `reports/vision_cad_feature_baseline_spatial_compare_20260106.json`
