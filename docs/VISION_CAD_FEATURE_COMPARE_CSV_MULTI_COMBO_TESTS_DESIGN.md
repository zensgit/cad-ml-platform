# Vision CAD Feature Compare CSV Multi Combo Tests Design

## Overview
Extend compare summary CSV coverage to validate multi-combo output when a grid
produces multiple threshold combinations.

## Test Coverage
- Runs a two-combo grid benchmark and verifies two CSV rows with `status=ok`.

## Tests
- `tests/unit/test_vision_cad_feature_benchmark_compare_csv.py`

## Command
```
pytest tests/unit/test_vision_cad_feature_benchmark_compare_csv.py -v
```
