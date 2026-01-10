# DEV_VISION_CAD_FEATURE_TUNING_COMPARE_RUN_50_VALIDATION_20260106

## Scope
Validate tuned compare outputs for a 50-sample run.

## Commands
- `python3 scripts/vision_cad_feature_benchmark.py --no-clients --input-dir data/dedup_report_train_local_version_profile_spatial_full_package/assets/images --max-samples 50 --threshold min_area=24 --threshold line_aspect=6 --threshold line_elongation=8 --threshold circle_fill_min=0.4 --threshold arc_fill_min=0.08 --output-json reports/vision_cad_feature_tuning_compare_20260106_50.json --output-csv reports/vision_cad_feature_tuning_compare_20260106_50.csv --compare-json reports/vision_cad_feature_grid_baseline_20260106_50.json --output-compare-csv reports/vision_cad_feature_tuning_compare_summary_20260106_50.csv`
- `python3 scripts/vision_cad_feature_compare_export.py --input-json reports/vision_cad_feature_tuning_compare_20260106_50.json --output-json reports/vision_cad_feature_tuning_compare_top_20260106_50.json --output-csv reports/vision_cad_feature_tuning_compare_top_20260106_50.csv --top-samples 5`

## Results
- `total_combos=1`
- Summary delta: `lines -156`, `circles -41`, `arcs 10`, `avg_components -3.74`
- Exported `top_samples=5`

## Outputs
- `reports/vision_cad_feature_tuning_compare_20260106_50.json`
- `reports/vision_cad_feature_tuning_compare_20260106_50.csv`
- `reports/vision_cad_feature_tuning_compare_summary_20260106_50.csv`
- `reports/vision_cad_feature_tuning_compare_top_20260106_50.json`
- `reports/vision_cad_feature_tuning_compare_top_20260106_50.csv`
