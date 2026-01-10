# Vision CAD Feature Compare Report Error Tests Design

## Overview
Extend compare report coverage to ensure missing comparison data fails fast
with a clear error.

## Tests
- `tests/unit/test_vision_cad_feature_compare_report.py`
  - Missing `comparison` block returns non-zero exit code.

## Validation
- `pytest tests/unit/test_vision_cad_feature_compare_report.py -v`
