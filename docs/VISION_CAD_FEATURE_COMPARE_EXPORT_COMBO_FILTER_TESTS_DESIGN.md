# Vision CAD Feature Compare Export Combo Filter Tests Design

## Overview
Extend compare export coverage to ensure the `--combo-index` filter returns a
single combo export and matching CSV rows.

## Test Coverage
- Validates JSON output only includes the selected combo index.
- Validates CSV output includes only the selected combo's top sample rows.

## Tests
- `tests/unit/test_vision_cad_feature_compare_export.py`

## Command
```
pytest tests/unit/test_vision_cad_feature_compare_export.py -v
```
