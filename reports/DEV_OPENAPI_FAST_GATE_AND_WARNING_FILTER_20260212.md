# DEV_OPENAPI_FAST_GATE_AND_WARNING_FILTER_20260212

## Summary

Hardened OpenAPI regression feedback by adding a dedicated CI fast gate and reducing
known non-actionable Pydantic deprecation warning noise in test output.

## Changes

- Updated `pytest.ini`:
  - Added a targeted warning filter for the known Pydantic v1/v2 migration warning:
    - `` `__fields__` attribute is deprecated, use `model_fields` instead ``

- Updated `.github/workflows/ci.yml`:
  - Added new job `openapi-fast` (Python 3.11) after `lint-type`.
  - `openapi-fast` runs `make validate-openapi` as an early, isolated API schema gate.
  - Added step summary output for the fast gate job status.
  - Updated `tests` job dependency to `needs: [lint-type, openapi-fast]`.

## Validation

- `make validate-openapi`
  - Result: `4 passed`
  - Note: no noisy Pydantic `__fields__` deprecation warning in output.

- `make validate-core-fast`
  - Result: passed
  - Evidence:
    - tolerance suite: `48 passed`
    - openapi/route suite: `4 passed`
    - service-mesh suite: `103 passed`
    - provider-core suite: `59 passed`
    - provider-contract suite: `4 passed, 20 deselected`

## Outcome

- OpenAPI contract regressions now fail faster in CI.
- Core-fast and local validation output is cleaner and easier to inspect.
