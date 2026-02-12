# DEV_PROVIDER_HEALTH_CONTRACT_COVERAGE_20260212

## Summary
- Added API contract coverage for provider health payloads to prevent regression of newly added plugin diagnostics fields.

## Changes
- Updated `tests/contract/test_api_contract.py`:
  - added `TestProviderHealthContracts::test_provider_health_endpoint_response_shape`.
  - added `TestProviderHealthContracts::test_health_payload_core_provider_plugin_summary_shape`.
- Contract assertions now explicitly validate:
  - `/api/v1/providers/health` includes `plugin_diagnostics` with counts, cache, and summary fields.
  - `/health` includes `config.core_providers.plugins.summary` with expected summary keys.

## Validation
- `pytest -q tests/contract/test_api_contract.py -k "provider_health_endpoint_response_shape or health_payload_core_provider_plugin_summary_shape"`
  - `2 passed, 20 deselected`
- `make test-provider-core`
  - `59 passed`

## Notes
- The contract checks are strict on required keys for diagnostics while remaining independent of specific provider names.
