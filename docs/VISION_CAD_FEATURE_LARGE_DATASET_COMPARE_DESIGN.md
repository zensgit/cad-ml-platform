# Vision CAD Feature Large Dataset Compare Design

## Overview
Establish a larger real-sample baseline and compare run for the CAD feature
benchmark to validate delta reporting on a wider dataset.

## Dataset
- Source: `data/dedup_report_train_local_version_profile_spatial_full_package/assets/images`
- Limit: `--max-samples 20`

## Commands
Baseline:
```
python3 scripts/vision_cad_feature_benchmark.py \
  --no-clients \
  --input-dir data/dedup_report_train_local_version_profile_spatial_full_package/assets/images \
  --max-samples 20 \
  --output-json reports/vision_cad_feature_baseline_spatial_20260106.json
```

Compare:
```
python3 scripts/vision_cad_feature_benchmark.py \
  --no-clients \
  --input-dir data/dedup_report_train_local_version_profile_spatial_full_package/assets/images \
  --max-samples 20 \
  --output-json reports/vision_cad_feature_baseline_spatial_compare_20260106.json \
  --compare-json reports/vision_cad_feature_baseline_spatial_20260106.json
```

## Outputs
- `reports/vision_cad_feature_baseline_spatial_20260106.json`
- `reports/vision_cad_feature_baseline_spatial_compare_20260106.json`
