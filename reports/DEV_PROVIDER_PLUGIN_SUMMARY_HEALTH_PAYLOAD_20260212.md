# DEV_PROVIDER_PLUGIN_SUMMARY_HEALTH_PAYLOAD_20260212

## Summary
- Added provider plugin summary aggregation to core provider registry snapshots.
- Upgraded health model typing for provider plugin payloads while keeping response compatibility.

## Changes
- Updated `src/core/providers/bootstrap.py`:
  - Added `_build_plugin_summary(status)`.
  - Added `plugins.summary` in registry snapshots with:
    - `configured_count`
    - `loaded_count`
    - `error_count`
    - `missing_registered_count`
    - `cache_reused`
    - `cache_reason`
    - `overall_status` (`ok`/`degraded`/`error`)
  - Hardened lazy snapshot bootstrap recovery when `_BOOTSTRAPPED=True` but registry is empty.
- Updated `src/api/health_models.py`:
  - Added typed models for plugin payload:
    - `HealthProviderPluginCache`
    - `HealthProviderPluginSummary`
    - `HealthProviderPlugins`
  - Switched `HealthConfigCoreProviders.plugins` to `Optional[HealthProviderPlugins]`.

## Validation
- `pytest -q tests/unit/test_provider_registry_plugins.py tests/unit/test_bootstrap_coverage.py tests/unit/test_provider_plugin_example_classifier.py tests/unit/test_provider_registry_bootstrap.py tests/unit/test_provider_framework.py tests/unit/test_provider_readiness.py tests/unit/test_health_utils_coverage.py tests/unit/test_health_hybrid_config.py tests/unit/test_provider_health_endpoint.py tests/unit/test_provider_check_metrics_exposed.py`
  - `57 passed`
- `make test-provider-core`
  - `57 passed`

## Notes
- The new summary is additive; existing plugin payload fields remain available.
