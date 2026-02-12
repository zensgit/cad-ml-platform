# DEV_PROVIDER_REGISTRY_OPENAPI_MODEL_20260212

## Summary

Promoted `/api/v1/providers/registry` response `registry` field from an untyped
`Dict[str, Any]` to the typed `HealthConfigCoreProviders` Pydantic model so the
OpenAPI schema reflects the provider registry snapshot contract.

## Changes

- Updated `src/api/v1/health.py`
  - `ProviderRegistryHealthResponse.registry` -> `HealthConfigCoreProviders`

- Updated `tests/contract/test_api_contract.py`
  - Added response shape contract for:
    - `GET /api/v1/providers/registry`
  - Added OpenAPI schema contract to assert `registry` is typed and contains
    expected properties.

## Validation

- `.venv/bin/python -m pytest tests/unit/test_health_hybrid_config.py tests/contract/test_api_contract.py -k provider_registry -v`
  - Result: `3 passed`

- `make openapi-snapshot-update`
  - Result: baseline regenerated (no semantic drift expected beyond typing)
  - Evidence: `paths=161`, `operations=166`

- `make validate-openapi`
  - Result: `5 passed`

- `make validate-core-fast`
  - Result: passed

## Outcome

The provider registry snapshot now has a first-class OpenAPI schema, which
reduces ambiguity for clients and improves contract drift detection.

