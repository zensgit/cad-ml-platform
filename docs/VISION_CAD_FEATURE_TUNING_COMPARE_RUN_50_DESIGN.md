# Vision CAD Feature Tuning Compare Run 50 Design

## Overview
Run the tuned (non-grid) compare workflow on 50 samples, capturing JSON/CSV
outputs and top sample deltas.

## Dataset
- Source: `data/dedup_report_train_local_version_profile_spatial_full_package/assets/images`
- Limit: `--max-samples 50`

## Threshold Overrides
- `min_area=24`
- `line_aspect=6`
- `line_elongation=8`
- `circle_fill_min=0.4`
- `arc_fill_min=0.08`

## Commands
```
python3 scripts/vision_cad_feature_benchmark.py \
  --no-clients \
  --input-dir data/dedup_report_train_local_version_profile_spatial_full_package/assets/images \
  --max-samples 50 \
  --threshold min_area=24 \
  --threshold line_aspect=6 \
  --threshold line_elongation=8 \
  --threshold circle_fill_min=0.4 \
  --threshold arc_fill_min=0.08 \
  --output-json reports/vision_cad_feature_tuning_compare_20260106_50.json \
  --output-csv reports/vision_cad_feature_tuning_compare_20260106_50.csv \
  --compare-json reports/vision_cad_feature_grid_baseline_20260106_50.json \
  --output-compare-csv reports/vision_cad_feature_tuning_compare_summary_20260106_50.csv

python3 scripts/vision_cad_feature_compare_export.py \
  --input-json reports/vision_cad_feature_tuning_compare_20260106_50.json \
  --output-json reports/vision_cad_feature_tuning_compare_top_20260106_50.json \
  --output-csv reports/vision_cad_feature_tuning_compare_top_20260106_50.csv \
  --top-samples 5
```

## Outputs
- `reports/vision_cad_feature_tuning_compare_20260106_50.json`
- `reports/vision_cad_feature_tuning_compare_20260106_50.csv`
- `reports/vision_cad_feature_tuning_compare_summary_20260106_50.csv`
- `reports/vision_cad_feature_tuning_compare_top_20260106_50.json`
- `reports/vision_cad_feature_tuning_compare_top_20260106_50.csv`
