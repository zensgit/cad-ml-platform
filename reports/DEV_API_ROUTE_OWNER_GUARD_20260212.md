# DEV_API_ROUTE_OWNER_GUARD_20260212

## Summary
- Strengthened API route guardrails to ensure critical `/api/v1/analyze/*` paths are owned by split routers (`drift.py` / `process.py`) and cannot be silently re-shadowed.

## Changes
- Updated `tests/unit/test_api_route_uniqueness.py`:
  - Added helper `_find_route_module(method, path)`.
  - Added `test_critical_analyze_paths_are_owned_by_split_routers` asserting module ownership for:
    - `GET /api/v1/analyze/drift` -> `src.api.v1.drift`
    - `POST /api/v1/analyze/drift/reset` -> `src.api.v1.drift`
    - `GET /api/v1/analyze/drift/baseline/status` -> `src.api.v1.drift`
    - `GET /api/v1/analyze/process/rules/audit` -> `src.api.v1.process`

## Validation
- `make validate-openapi`
  - result: `3 passed`
- `make validate-core-fast`
  - result: `ISO286 validators OK`, `48 passed`, `3 passed`, `103 passed`, `59 passed`

## Notes
- This guard complements:
  - OpenAPI `operationId` uniqueness checks;
  - runtime duplicate `(method, path)` checks.
