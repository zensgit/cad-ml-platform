# DEV_VISION_CAD_FEATURE_GRID_COMPARE_EXPORT_RUN_50_VALIDATION_20260106

## Scope
Validate grid compare/export outputs for a 50-sample run.

## Commands
- `python3 scripts/vision_cad_feature_benchmark.py --no-clients --input-dir data/dedup_report_train_local_version_profile_spatial_full_package/assets/images --max-samples 50 --threshold-file examples/cad_feature_thresholds.json --output-json reports/vision_cad_feature_grid_baseline_20260106_50.json`
- `python3 scripts/vision_cad_feature_benchmark.py --no-clients --input-dir data/dedup_report_train_local_version_profile_spatial_full_package/assets/images --max-samples 50 --threshold-file examples/cad_feature_thresholds.json --threshold min_area=24 --output-json reports/vision_cad_feature_grid_compare_20260106_50.json --compare-json reports/vision_cad_feature_grid_baseline_20260106_50.json --output-compare-csv reports/vision_cad_feature_grid_compare_summary_20260106_50.csv`
- `python3 scripts/vision_cad_feature_compare_export.py --input-json reports/vision_cad_feature_grid_compare_20260106_50.json --output-json reports/vision_cad_feature_grid_compare_top_20260106_50.json --output-csv reports/vision_cad_feature_grid_compare_top_20260106_50.csv --top-samples 5`

## Results
- `total_combos=4`
- Combo 1 summary delta: `lines -141`, `circles -35`, `arcs 0`, `avg_components -3.52`
- Exported `top_samples=5` per combo

## Outputs
- `reports/vision_cad_feature_grid_baseline_20260106_50.json`
- `reports/vision_cad_feature_grid_compare_20260106_50.json`
- `reports/vision_cad_feature_grid_compare_summary_20260106_50.csv`
- `reports/vision_cad_feature_grid_compare_top_20260106_50.json`
- `reports/vision_cad_feature_grid_compare_top_20260106_50.csv`
