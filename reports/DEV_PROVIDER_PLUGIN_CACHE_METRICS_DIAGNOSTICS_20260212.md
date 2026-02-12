# DEV_PROVIDER_PLUGIN_CACHE_METRICS_DIAGNOSTICS_20260212

## Summary
- Added plugin-bootstrap observability metrics for the provider framework.
- Enhanced plugin cache diagnostics in registry snapshots to make cache-hit vs reload reasons explicit.

## Changes
- Updated `src/utils/metrics.py`
  - Added `core_provider_plugin_bootstrap_total{result}`
  - Added `core_provider_plugin_bootstrap_duration_seconds{result}`
  - Added gauges:
    - `core_provider_plugin_configured`
    - `core_provider_plugin_loaded`
    - `core_provider_plugin_errors`
- Updated `src/core/providers/bootstrap.py`
  - Added plugin cache diagnostics under `plugins.cache`:
    - `reused`
    - `reason`
    - `checked_at`
    - `missing_registered`
  - Added integrity diagnostics for cached plugin registration state.
  - Added automatic metric emission for:
    - cache hit
    - reload ok / reload partial
    - strict error
- Updated `tests/unit/test_provider_registry_plugins.py`
  - Added assertions for cache metadata on first load and post-reset reload.
  - Added new regression test for cache-hit behavior when registry remains intact.

## Validation
- `pytest -q tests/unit/test_provider_registry_plugins.py tests/unit/test_bootstrap_coverage.py tests/unit/test_provider_plugin_example_classifier.py`
  - `17 passed`
- `pytest -q tests/unit/test_provider_registry_bootstrap.py tests/unit/test_provider_framework.py tests/unit/test_provider_readiness.py tests/unit/test_health_utils_coverage.py`
  - `33 passed`

## Notes
- This is backward-compatible: no public API contract change.
- Health payload now carries richer plugin cache diagnostics via `config.core_providers.plugins.cache`.
