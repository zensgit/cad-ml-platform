# Vision CAD Feature Threshold File YAML Run Design

## Overview
Validate YAML threshold file parsing using the benchmark script.

## Files
- `reports/vision_cad_feature_threshold_file_20260106.yaml`
- `reports/vision_cad_feature_threshold_file_yaml_run_20260106.json`

## Command
```
python3 scripts/vision_cad_feature_benchmark.py \
  --no-clients \
  --max-samples 1 \
  --threshold-file reports/vision_cad_feature_threshold_file_20260106.yaml \
  --output-json reports/vision_cad_feature_threshold_file_yaml_run_20260106.json
```
