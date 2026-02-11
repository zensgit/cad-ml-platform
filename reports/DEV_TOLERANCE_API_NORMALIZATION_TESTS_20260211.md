# DEV_TOLERANCE_API_NORMALIZATION_TESTS_20260211

## Summary
- Added regression coverage for tolerance fit-code normalization behavior in `src/api/v1/tolerance.py`.
- Locked mixed-case and whitespace handling to prevent endpoint drift.

## Changes
- Added `tests/unit/test_tolerance_api_normalization.py`
  - `_normalize_fit_code` normalization behavior
  - endpoint-level behavior for mixed-case + whitespace fit codes

## Validation
- `pytest -q tests/unit/test_tolerance_api_normalization.py`
  - Result: `6 passed`
- `pytest -q tests/integration/test_tolerance_api.py`
  - Result: `3 passed`

