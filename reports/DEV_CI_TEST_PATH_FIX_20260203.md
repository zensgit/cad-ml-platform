# DEV_CI_TEST_PATH_FIX_20260203

## Summary
- Ensured repo root is inserted into `sys.path` during pytest collection to resolve `src.core.*` imports in CI.
- This targets the remaining CI failure from `tests/unit/test_enterprise_p20_p23.py` (module import error).

## Changes
- `tests/conftest.py`
  - Inserted repo root into `sys.path` before test collection.

## Validation
- Not rerun in CI yet; will re-dispatch `ci.yml` after commit.
