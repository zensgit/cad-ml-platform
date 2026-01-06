# Vision CAD Feature Compare CSV Missing Baseline Tests Design

## Overview
Extend compare summary CSV coverage to ensure missing baseline combos are
represented with a `missing_baseline` status.

## Test Coverage
- Validate that a comparison against an empty baseline yields a CSV row with
  `status=missing_baseline`.

## Tests
- `tests/unit/test_vision_cad_feature_benchmark_compare_csv.py`

## Command
```
pytest tests/unit/test_vision_cad_feature_benchmark_compare_csv.py -v
```
