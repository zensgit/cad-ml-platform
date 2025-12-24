# DEV_DRIFT_AUTO_REFRESH_STARTUP_20251224

## Scope
- Enable drift auto-refresh tests and verify startup-trigger metrics.
- Fix stale flag logic to treat zero-age baselines as non-stale.

## Changes
- `src/api/v1/analyze.py`
  - Added auto-refresh refresh handling with stale triggers and startup refresh metrics.
  - Fixed stale flag evaluation for zero-age baselines.
- `src/api/v1/drift.py`
  - Fixed stale flag evaluation for zero-age baselines.
- `tests/unit/test_drift_auto_refresh.py`
  - Removed skips for auto-refresh scenarios.
- `tests/unit/test_drift_startup_trigger.py`
  - Implemented startup-trigger metric assertions.

## Validation
- Command: `.venv/bin/python -m pytest tests/unit/test_drift_auto_refresh.py tests/unit/test_drift_startup_trigger.py -v`
  - Result: 7 passed, 1 skipped.
