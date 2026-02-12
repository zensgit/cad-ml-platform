# DEV_PROVIDER_PLUGIN_RELOAD_RESILIENCE_20260212

## Summary
- Improved provider plugin bootstrap resilience when `ProviderRegistry` is cleared while plugin env config remains unchanged.
- Added plugin cache integrity checks so plugin bootstrap can recover missing plugin-registered providers automatically.

## Problem
- `bootstrap_core_provider_plugins()` previously reused cached plugin status based only on env config (`CORE_PROVIDER_PLUGINS`, strict mode).
- After `ProviderRegistry.clear()`, plugin-registered providers could be missing while cache still reported loaded, causing plugin bootstrap to be skipped.

## Changes
- Updated `src/core/providers/bootstrap.py`:
  - Added plugin status field `registered` to track provider IDs added per plugin token.
  - Added `_snapshot_provider_ids()` helper to compute registry diff around each plugin load.
  - Added `_plugins_registry_intact()` guard to validate cached plugin registrations still exist.
  - Cache now reuses status only when config matches **and** registered providers are still present.
  - Kept non-strict behavior and strict error raising semantics unchanged.
- Updated `tests/unit/test_provider_registry_plugins.py`:
  - Added autouse fixture to reset plugin cache state for deterministic tests.
  - Added regression test `test_bootstrap_plugin_reloads_after_registry_clear`.

## Validation
- `pytest -q tests/unit/test_provider_registry_plugins.py tests/unit/test_bootstrap_coverage.py tests/unit/test_provider_plugin_example_classifier.py`
  - `16 passed`
- `pytest -q tests/unit/test_provider_registry_bootstrap.py tests/unit/test_provider_framework.py tests/unit/test_provider_readiness.py`
  - `15 passed`

## Impact
- Provider framework becomes robust to runtime/test registry resets without requiring manual `reset_core_provider_plugins_state()`.
- No API contract change; only bootstrap cache behavior is hardened.
