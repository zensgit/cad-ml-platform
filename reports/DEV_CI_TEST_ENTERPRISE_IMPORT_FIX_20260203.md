# DEV_CI_TEST_ENTERPRISE_IMPORT_FIX_20260203

## Summary
- CI still resolved a non-local `src` package when collecting `tests/unit/test_enterprise_p20_p23.py`.
- Added a local repo path insert and forced re-import inside the test module to ensure `src.core.cache` resolves.

## Changes
- `tests/unit/test_enterprise_p20_p23.py`
  - Inserted repo root into `sys.path`, cleared `sys.modules["src"]`, and invalidated caches before imports.

## Validation
- `python3 -m pytest tests/unit/test_enterprise_p20_p23.py -q`
