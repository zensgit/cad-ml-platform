# DEV_PROVIDER_HEALTH_OPENAPI_CONTRACT_20260212

## Summary
- Added OpenAPI-level contract checks for provider health payload fields to guard against schema drift.

## Changes
- Updated `tests/contract/test_api_contract.py`:
  - Added `_openapi_schema()` helper (cached OpenAPI fetch).
  - Added `_resolve_schema_ref()` helper (component schema dereference).
  - Added `TestProviderHealthContracts::test_provider_health_openapi_schema_contains_plugin_diagnostics`.
  - Added `TestProviderHealthContracts::test_health_openapi_schema_contains_core_provider_plugin_summary`.
- New contract coverage verifies:
  - `/api/v1/providers/health` OpenAPI response schema includes `plugin_diagnostics`.
  - OpenAPI components expose provider plugin summary schemas:
    - `HealthConfigCoreProviders.properties.plugins`
    - `HealthProviderPlugins.properties.summary`
    - `HealthProviderPluginSummary` required summary fields.

## Validation
- `pytest -q tests/contract/test_api_contract.py -k "provider_health or core_provider_plugin_summary or openapi_schema_contains_plugin_diagnostics or openapi_schema_contains_core_provider_plugin_summary"`
  - `4 passed, 20 deselected`
- `make test-provider-core`
  - `59 passed`

## Notes
- FastAPI currently emits duplicate operation-id warnings for drift/process endpoints during OpenAPI generation; this does not affect test correctness but should be cleaned up separately.
