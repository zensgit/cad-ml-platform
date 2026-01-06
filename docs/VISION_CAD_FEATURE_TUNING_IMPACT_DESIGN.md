# Vision CAD Feature Tuning Impact Design

## Overview
Run a benchmark with stricter heuristic thresholds and compare against the
baseline to quantify tuning impact on line/circle/arc counts.

## Dataset
- Source: `data/dedup_report_train_local_version_profile_spatial_full_package/assets/images`
- Limit: `--max-samples 20`

## Threshold Overrides
- `min_area=24`
- `line_aspect=6`
- `line_elongation=8`
- `circle_fill_min=0.4`
- `arc_fill_min=0.08`

## Command
```
python3 scripts/vision_cad_feature_benchmark.py \
  --no-clients \
  --input-dir data/dedup_report_train_local_version_profile_spatial_full_package/assets/images \
  --max-samples 20 \
  --threshold min_area=24 \
  --threshold line_aspect=6 \
  --threshold line_elongation=8 \
  --threshold circle_fill_min=0.4 \
  --threshold arc_fill_min=0.08 \
  --output-json reports/vision_cad_feature_tuning_compare_20260106.json \
  --output-csv reports/vision_cad_feature_tuning_compare_20260106.csv \
  --compare-json reports/vision_cad_feature_baseline_spatial_20260106.json
```

## Outputs
- `reports/vision_cad_feature_tuning_compare_20260106.json`
- `reports/vision_cad_feature_tuning_compare_20260106.csv`
