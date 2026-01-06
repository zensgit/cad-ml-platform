# Vision CAD Feature Compare Export Invalid Index Tests Design

## Overview
Extend compare export coverage to ensure invalid combo index values (< 1) are
rejected.

## Test Coverage
- Validates that `--combo-index 0` exits with an error message.

## Tests
- `tests/unit/test_vision_cad_feature_compare_export.py`

## Command
```
pytest tests/unit/test_vision_cad_feature_compare_export.py -v
```
