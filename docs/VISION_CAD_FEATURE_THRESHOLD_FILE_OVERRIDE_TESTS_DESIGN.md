# Vision CAD Feature Threshold File Override Tests Design

## Overview
Extend threshold-file unit coverage to ensure CLI overrides take precedence over
file-provided thresholds.

## Tests
- `tests/unit/test_vision_cad_feature_benchmark_threshold_file.py`
  - CLI `--threshold min_area=24` overrides file value.

## Validation
- `pytest tests/unit/test_vision_cad_feature_benchmark_threshold_file.py -v`
