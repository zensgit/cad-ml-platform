# DEV_VISION_CAD_FEATURE_TUNING_IMPACT_VALIDATION_20260106

## Scope
Validate benchmark impact reporting with stricter heuristic thresholds.

## Command
- `python3 scripts/vision_cad_feature_benchmark.py --no-clients --input-dir data/dedup_report_train_local_version_profile_spatial_full_package/assets/images --max-samples 20 --threshold min_area=24 --threshold line_aspect=6 --threshold line_elongation=8 --threshold circle_fill_min=0.4 --threshold arc_fill_min=0.08 --output-json reports/vision_cad_feature_tuning_compare_20260106.json --output-csv reports/vision_cad_feature_tuning_compare_20260106.csv --compare-json reports/vision_cad_feature_baseline_spatial_20260106.json`

## Results
- `total_samples=20`, `total_combos=1`
- Summary delta: `lines -82`, `circles -21`, `arcs +3`, `avg_components -5.0`

## Outputs
- `reports/vision_cad_feature_tuning_compare_20260106.json`
- `reports/vision_cad_feature_tuning_compare_20260106.csv`
