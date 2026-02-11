# DEV_MAKE_TEST_TOLERANCE_20260211

## Summary
- Added a dedicated Makefile target to execute tolerance knowledge tests with a single command.

## Changes
- Updated `Makefile`
  - Added `.PHONY` entry: `test-tolerance`
  - Added target: `make test-tolerance`
  - Scope:
    - `tests/unit/knowledge/test_tolerance.py`
    - `tests/unit/test_tolerance_fundamental_deviation.py`
    - `tests/unit/test_tolerance_limit_deviations.py`
    - `tests/unit/test_tolerance_api_normalization.py`
    - `tests/integration/test_tolerance_api.py`

## Validation
- `make test-tolerance`
  - Result: `44 passed`

