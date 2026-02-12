# DEV_PROVIDER_READINESS_ERROR_SANITIZATION_20260212

## Goal
Keep `/ready` (provider readiness) error strings safe for client display and log
aggregation by ensuring they are single-line and bounded in size.

## Change
- Updated `src/core/providers/readiness.py`:
  - sanitize provider readiness errors via `sanitize_single_line_text`
  - sanitize `init_error: ...` path as well
- Added unit coverage in `tests/unit/test_provider_readiness.py` to assert:
  - no `\\n` in `ProviderReadinessItem.error`
  - error length bounded to `<= 300`

## Validation
- `.venv/bin/python -m pytest tests/unit/test_provider_readiness.py -v`
  - Result: `3 passed`
- `make validate-core-fast`
  - Result: passed

## Files Changed
- `src/core/providers/readiness.py`
- `tests/unit/test_provider_readiness.py`

