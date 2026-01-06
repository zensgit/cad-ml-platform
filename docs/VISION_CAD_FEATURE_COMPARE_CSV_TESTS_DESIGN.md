# Vision CAD Feature Compare CSV Tests Design

## Overview
Add unit coverage for the benchmark compare summary CSV output and its baseline
requirement.

## Test Coverage
- Validate that `--output-compare-csv` writes the expected header and a row for
  a single combo.
- Validate that `--output-compare-csv` fails without `--compare-json`.

## Tests
- `tests/unit/test_vision_cad_feature_benchmark_compare_csv.py`

## Command
```
pytest tests/unit/test_vision_cad_feature_benchmark_compare_csv.py -v
```
