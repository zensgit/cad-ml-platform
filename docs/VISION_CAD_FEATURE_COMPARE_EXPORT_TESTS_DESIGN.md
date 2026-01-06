# Vision CAD Feature Compare Export Tests Design

## Overview
Add unit coverage for the compare export utility that produces JSON/CSV outputs
from benchmark comparison payloads.

## Test Coverage
- Validates JSON and CSV outputs, including top-sample selection ordering.
- Validates error handling for out-of-range `--combo-index`.

## Tests
- `tests/unit/test_vision_cad_feature_compare_export.py`

## Command
```
pytest tests/unit/test_vision_cad_feature_compare_export.py -v
```
