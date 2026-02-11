# DEV_PROVIDER_HEALTH_PLUGIN_DIAGNOSTICS_20260212

## Summary
- Extended provider health endpoint response with plugin diagnostics to align registry/health observability.

## Changes
- Updated `src/api/v1/health.py`:
  - Added `plugin_diagnostics` field to `ProviderHealthResponse`.
  - `provider_health()` now includes plugin diagnostics from `get_core_provider_registry_snapshot(lazy_bootstrap=False)`.
  - Diagnostics include:
    - `summary`
    - `cache`
    - `configured_count`
    - `loaded_count`
    - `error_count`
- Updated tests:
  - `tests/unit/test_provider_health_endpoint.py` now validates `plugin_diagnostics` presence and key values.

## Validation
- `pytest -q tests/unit/test_provider_health_endpoint.py tests/unit/test_provider_check_metrics_exposed.py tests/unit/test_health_hybrid_config.py`
  - Included in full provider core run and passed.
- `make test-provider-core`
  - `57 passed`

## Notes
- Endpoint contract is backward compatible (`plugin_diagnostics` is optional and additive).
