# DEV_CI_TEST_IMPORT_OVERRIDE_20260203

## Summary
- CI still resolved `src` from a non-local package, so `src.core.cache` remained missing.
- Forced local `src` to load by clearing any preloaded `src` module and re-importing after injecting repo root.

## Changes
- `tests/conftest.py`
  - Clears `sys.modules["src"]`, invalidates import caches, and imports local `src` when `src/` exists.

## Validation
- Not rerun in CI yet; will re-dispatch `ci.yml` after commit.
