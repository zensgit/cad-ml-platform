# DEV_PROVIDER_PLUGIN_DEVKIT_20260210

## Summary
- Added a provider plugin developer kit to integrate experimental providers via `CORE_PROVIDER_PLUGINS` with minimal merge risk.

## Changes
- Docs:
  - Added plugin guide: `docs/PROVIDER_PLUGIN_GUIDE.md`
  - Linked from provider framework guide: `docs/PROVIDER_FRAMEWORK.md`
- Example plugin module:
  - Added `src/core/provider_plugins/example_classifier.py` (registers `classifier/example_rules`)
  - Added `src/core/provider_plugins/__init__.py`
- Test support:
  - Added `reset_core_provider_plugins_state()` in `src/core/providers/bootstrap.py` to reset plugin cache in tests that clear the registry.
  - Added unit coverage: `tests/unit/test_provider_plugin_example_classifier.py`

## Validation
- `pytest -q tests/unit/test_provider_plugin_example_classifier.py tests/unit/test_provider_registry_plugins.py`
  - Result: pass

