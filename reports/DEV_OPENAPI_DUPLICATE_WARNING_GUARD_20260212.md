# DEV_OPENAPI_DUPLICATE_WARNING_GUARD_20260212

## Summary

Added an explicit regression guard to ensure OpenAPI schema generation does not emit
`Duplicate Operation ID` warnings.

## Change

- Updated `tests/contract/test_openapi_operation_ids.py`:
  - Added `test_openapi_generation_has_no_duplicate_operation_id_warnings`.
  - Test captures warnings during `/openapi.json` generation and fails if any warning message
    contains `Duplicate Operation ID`.

## Validation

- `make validate-openapi`
  - Result: `4 passed` (with existing Pydantic deprecation warnings, no duplicate operation-id warnings)
- `make validate-core-fast`
  - Result: passed
  - Evidence:
    - tolerance suite: `48 passed`
    - openapi/route suite: `4 passed`
    - service-mesh suite: `103 passed`
    - provider-core suite: `59 passed`
    - provider-contract suite: `4 passed, 20 deselected`

## Outcome

OpenAPI warning regressions are now covered by contract tests and remain green in the core-fast
validation gate.
