# Vision CAD Feature Grid Compare Export Run 100 Design

## Overview
Run the grid baseline/compare/export workflow on 100 samples to evaluate delta
stability on a larger dataset.

## Dataset
- Source: `data/dedup_report_train_local_version_profile_spatial_full_package/assets/images`
- Limit: `--max-samples 100`

## Threshold File
- `examples/cad_feature_thresholds.json`

## Override
- `min_area=24` for the compare run.

## Commands
```
# Baseline grid
python3 scripts/vision_cad_feature_benchmark.py \
  --no-clients \
  --input-dir data/dedup_report_train_local_version_profile_spatial_full_package/assets/images \
  --max-samples 100 \
  --threshold-file examples/cad_feature_thresholds.json \
  --output-json reports/vision_cad_feature_grid_baseline_20260106_100.json

# Compare grid
python3 scripts/vision_cad_feature_benchmark.py \
  --no-clients \
  --input-dir data/dedup_report_train_local_version_profile_spatial_full_package/assets/images \
  --max-samples 100 \
  --threshold-file examples/cad_feature_thresholds.json \
  --threshold min_area=24 \
  --output-json reports/vision_cad_feature_grid_compare_20260106_100.json \
  --compare-json reports/vision_cad_feature_grid_baseline_20260106_100.json \
  --output-compare-csv reports/vision_cad_feature_grid_compare_summary_20260106_100.csv

# Export top sample deltas
python3 scripts/vision_cad_feature_compare_export.py \
  --input-json reports/vision_cad_feature_grid_compare_20260106_100.json \
  --output-json reports/vision_cad_feature_grid_compare_top_20260106_100.json \
  --output-csv reports/vision_cad_feature_grid_compare_top_20260106_100.csv \
  --top-samples 5
```

## Outputs
- `reports/vision_cad_feature_grid_baseline_20260106_100.json`
- `reports/vision_cad_feature_grid_compare_20260106_100.json`
- `reports/vision_cad_feature_grid_compare_summary_20260106_100.csv`
- `reports/vision_cad_feature_grid_compare_top_20260106_100.json`
- `reports/vision_cad_feature_grid_compare_top_20260106_100.csv`
