# Vision CAD Feature Compare Export Missing Baseline Tests Design

## Overview
Extend compare export coverage to ensure missing baseline combos are emitted
with `missing_baseline` status and blank sample rows in CSV output.

## Test Coverage
- Validate JSON output marks missing baseline combos with empty sample lists.
- Validate CSV output includes a row with `status=missing_baseline`.

## Tests
- `tests/unit/test_vision_cad_feature_compare_export.py`

## Command
```
pytest tests/unit/test_vision_cad_feature_compare_export.py -v
```
