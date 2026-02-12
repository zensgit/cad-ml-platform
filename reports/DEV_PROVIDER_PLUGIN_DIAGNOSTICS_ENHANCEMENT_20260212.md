# DEV_PROVIDER_PLUGIN_DIAGNOSTICS_ENHANCEMENT_20260212

## Summary

Enhanced provider plugin diagnostics in `/api/v1/providers/health` to expose a
small, bounded sample of plugin bootstrap errors and registered provider IDs.
This makes CI and runtime debugging easier without requiring log access.

## Changes

- Updated `src/api/v1/health.py`
  - Extended `plugin_diagnostics` with:
    - `errors_sample` (max 10): list of `{plugin, error}`
    - `errors_truncated`: whether more errors exist beyond the sample
    - `registered_count`: total number of registered provider IDs across plugins
    - `registered_sample` (max 25): sample of provider IDs registered by plugins

- Updated `tests/contract/test_api_contract.py`
  - Extended provider health response contract assertions to require:
    - `errors_sample`, `errors_truncated`, `registered_count`, `registered_sample`

- Updated `tests/unit/test_provider_health_endpoint.py`
  - Asserted the new diagnostic fields are present and stable for the unit
    health endpoint test.

## Validation

- `make validate-core-fast`
  - Result: passed
  - Evidence:
    - tolerance suite: `48 passed`
    - openapi/route suite: `5 passed`
    - service-mesh suite: `103 passed`
    - provider-core suite: `60 passed`
    - provider-contract suite: `4 passed, 20 deselected`

## Outcome

Provider plugin issues are now visible directly from the health endpoint in a
bounded, API-safe form, reducing time-to-root-cause in CI and production.

