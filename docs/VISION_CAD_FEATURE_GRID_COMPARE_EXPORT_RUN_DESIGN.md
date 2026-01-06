# Vision CAD Feature Grid Compare Export Run Design

## Overview
Run a multi-combo grid benchmark on real CAD images, compare against a baseline
with stricter `min_area`, and export summary CSV plus top-sample deltas.

## Dataset
- Source: `data/dedup_report_train_local_version_profile_spatial_full_package/assets/images`
- Limit: `--max-samples 20`

## Threshold File
- `examples/cad_feature_thresholds.json`
- Grid: `arc_fill_min`, `arc_fill_max` (4 combos)

## Override
- `min_area=24` for the compare run.

## Commands
```
# Baseline grid
python3 scripts/vision_cad_feature_benchmark.py \
  --no-clients \
  --input-dir data/dedup_report_train_local_version_profile_spatial_full_package/assets/images \
  --max-samples 20 \
  --threshold-file examples/cad_feature_thresholds.json \
  --output-json reports/vision_cad_feature_grid_baseline_20260106.json

# Compare grid
python3 scripts/vision_cad_feature_benchmark.py \
  --no-clients \
  --input-dir data/dedup_report_train_local_version_profile_spatial_full_package/assets/images \
  --max-samples 20 \
  --threshold-file examples/cad_feature_thresholds.json \
  --threshold min_area=24 \
  --output-json reports/vision_cad_feature_grid_compare_20260106.json \
  --compare-json reports/vision_cad_feature_grid_baseline_20260106.json \
  --output-compare-csv reports/vision_cad_feature_grid_compare_summary_20260106.csv

# Export top sample deltas
python3 scripts/vision_cad_feature_compare_export.py \
  --input-json reports/vision_cad_feature_grid_compare_20260106.json \
  --output-json reports/vision_cad_feature_grid_compare_top_20260106.json \
  --output-csv reports/vision_cad_feature_grid_compare_top_20260106.csv \
  --top-samples 5
```

## Outputs
- `reports/vision_cad_feature_grid_baseline_20260106.json`
- `reports/vision_cad_feature_grid_compare_20260106.json`
- `reports/vision_cad_feature_grid_compare_summary_20260106.csv`
- `reports/vision_cad_feature_grid_compare_top_20260106.json`
- `reports/vision_cad_feature_grid_compare_top_20260106.csv`
