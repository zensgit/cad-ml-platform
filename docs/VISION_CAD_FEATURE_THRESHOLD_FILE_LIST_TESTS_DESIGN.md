# Vision CAD Feature Threshold File List Tests Design

## Overview
Extend threshold-file unit coverage to ensure list payloads are treated as
explicit variants.

## Tests
- `tests/unit/test_vision_cad_feature_benchmark_threshold_file.py`
  - List payload produces the expected number of combos.

## Validation
- `pytest tests/unit/test_vision_cad_feature_benchmark_threshold_file.py -v`
