# DEV_OPENAPI_OPERATION_ID_DEDUP_20260212

## Summary
- Removed OpenAPI `Duplicate Operation ID` warnings caused by duplicated drift/process endpoints across split routers.
- Added automated guardrail test for global `operationId` uniqueness.
- Integrated OpenAPI uniqueness check into core fast validation and CI summaries.

## Changes
- Route metadata hardening:
  - `src/api/v1/process.py`: added explicit `operation_id="process_rules_audit_v1"`.
  - `src/api/v1/drift.py`: added explicit operation IDs for drift endpoints (`drift_status_v1`, `drift_reset_v1`, `drift_baseline_status_v1`, `drift_baseline_export_v1`, `drift_baseline_import_v1`).
  - `src/api/v1/analyze.py`: marked duplicate legacy routes as `include_in_schema=False` and assigned legacy operation IDs:
    - `process_rules_audit_legacy`
    - `drift_status_legacy`
    - `drift_reset_legacy`
    - `drift_baseline_status_legacy`
- Contract test:
  - added `tests/contract/test_openapi_operation_ids.py` to assert OpenAPI operation IDs are globally unique.
- Build/CI integration:
  - `Makefile`: added `validate-openapi` target and included it in `validate-core-fast`.
  - `.github/workflows/ci.yml`: core-fast summary now parses and displays `openapi-contract suite`.
  - `.github/workflows/ci-tiered-tests.yml`: same summary enhancement.

## Validation
- Duplicate warning check:
  - `python3` probe with `warnings.catch_warnings` over `/openapi.json`
  - result: `duplicate_operation_id_warnings 0`
- OpenAPI uniqueness gate:
  - `make validate-openapi`
  - result: `1 passed`
- Core fast regression:
  - `make validate-core-fast`
  - result: `ISO286 validators OK`, `48 passed`, `1 passed`, `103 passed`, `59 passed`

## Notes
- Existing pydantic deprecation warnings remain non-blocking and unrelated to operationId collisions.
