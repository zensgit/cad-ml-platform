# Vision CAD Feature Threshold File Tests Design

## Overview
Add unit coverage for the benchmark CLI `--threshold-file` option to ensure grid
and variant payloads are processed correctly.

## Tests
- `tests/unit/test_vision_cad_feature_benchmark_threshold_file.py`
  - Grid payload produces 4 combos.
  - Variants payload produces explicit combo count.

## Validation
- `pytest tests/unit/test_vision_cad_feature_benchmark_threshold_file.py -v`
