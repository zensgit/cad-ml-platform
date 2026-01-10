# Vision CAD Feature Compare Export Stdout Tests Design

## Overview
Extend compare export coverage to validate JSON output emitted to stdout when no
output files are specified.

## Test Coverage
- Runs the export script with only `--input-json` and asserts JSON is printed.

## Tests
- `tests/unit/test_vision_cad_feature_compare_export.py`

## Command
```
pytest tests/unit/test_vision_cad_feature_compare_export.py -v
```
