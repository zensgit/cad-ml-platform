# DEV_API_ROUTE_RUNTIME_DEDUP_20260212

## Summary
- Removed runtime duplicate method/path route registrations under `/api/v1/analyze/*` for drift and process audit endpoints.
- Added a route uniqueness regression test to prevent shadow-route regressions.

## Changes
- `src/api/v1/analyze.py`:
  - Removed route decorators from legacy functions:
    - `process_rules_audit`
    - `drift_status`
    - `drift_reset`
    - `drift_baseline_status`
  - Result: these legacy functions are no longer registered as API routes.
- `tests/unit/test_api_route_uniqueness.py`:
  - New test `test_api_routes_have_no_duplicate_method_path_pairs`.
  - Fails on any duplicate `(HTTP method, path)` pair in FastAPI route table.
- `Makefile`:
  - `validate-openapi` now runs both:
    - `tests/contract/test_openapi_operation_ids.py`
    - `tests/unit/test_api_route_uniqueness.py`

## Validation
- Runtime duplicate check probe:
  - result: `duplicate_method_path_count 0`
- `make validate-openapi`
  - result: `2 passed`
- `make validate-core-fast`
  - result: `ISO286 validators OK`, `48 passed`, `2 passed`, `103 passed`, `59 passed`
- Drift endpoint regression:
  - `pytest -q tests/unit/test_drift_endpoint.py tests/unit/test_drift_startup_trigger.py`
  - result: `2 passed`

## Notes
- OpenAPI operation-id de-duplication from the previous step remains effective (`duplicate_operation_id_warnings 0`).
