# DEV_CORE_PROVIDER_PLUGIN_BOOTSTRAP_20260210

## Summary
- Added a best-effort plugin hook for the core provider framework so experimental providers can be registered via environment variable without modifying the built-in bootstrap list.

## Implementation
- `src/core/providers/bootstrap.py`
  - New env:
    - `CORE_PROVIDER_PLUGINS`: comma/space separated plugin tokens
      - `pkg.module` (import-only; module expected to self-register providers)
      - `pkg.module:bootstrap` (import + call bootstrap function)
    - `CORE_PROVIDER_PLUGINS_STRICT`: when `true`, plugin import/call errors raise
  - Snapshot now includes `plugins` metadata:
    - `enabled`, `strict`, `configured`, `loaded`, `errors`

## Ops Documentation
- `.env.example` now documents the plugin env vars in the provider registry section.

## Validation
- `pytest -q tests/unit/test_provider_registry_plugins.py tests/unit/test_provider_registry_bootstrap.py`
  - Result: pass

## Notes
- Plugins should be idempotent (use `ProviderRegistry.exists(...)` guards) because bootstrap may run multiple times.

