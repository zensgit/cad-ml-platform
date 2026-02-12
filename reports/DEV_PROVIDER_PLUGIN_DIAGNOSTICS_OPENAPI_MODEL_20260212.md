# DEV_PROVIDER_PLUGIN_DIAGNOSTICS_OPENAPI_MODEL_20260212

## Summary

Promoted `/api/v1/providers/health` `plugin_diagnostics` from an untyped
`Dict[str, Any]` to an explicit Pydantic model so the OpenAPI schema accurately
describes the diagnostics fields, and the OpenAPI snapshot gate can detect
contract drift.

## Changes

- Updated `src/api/v1/health.py`
  - Added response models:
    - `ProviderPluginErrorItem`
    - `ProviderPluginDiagnostics`
  - Updated `ProviderHealthResponse.plugin_diagnostics` to:
    - `Optional[ProviderPluginDiagnostics]`
  - Reused existing health models for nested fields:
    - `HealthProviderPluginSummary`
    - `HealthProviderPluginCache`

- Updated OpenAPI snapshot baseline
  - `config/openapi_schema_snapshot.json` regenerated via:
    - `make openapi-snapshot-update`

## Validation

- `.venv/bin/python -m pytest tests/unit/test_provider_health_endpoint.py tests/contract/test_api_contract.py -k provider_health -v`
  - Result: `4 passed`

- `make openapi-snapshot-update`
  - Result: baseline regenerated
  - Evidence: `paths=161`, `operations=166`

- `make validate-openapi`
  - Result: `5 passed`

- `make validate-core-fast`
  - Result: passed
  - Evidence:
    - tolerance suite: `48 passed`
    - openapi/route suite: `5 passed`
    - service-mesh suite: `103 passed`
    - provider-core suite: `60 passed`
    - provider-contract suite: `4 passed, 20 deselected`

## Outcome

The health endpoint diagnostics are now a first-class OpenAPI contract surface,
with drift detection enforced by the OpenAPI snapshot gate.

