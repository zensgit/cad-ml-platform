# DEV_PROVIDER_HEALTH_ERROR_SANITIZATION_20260212

## Goal
Reduce the risk of leaking multi-line / oversized exception context in public
health payloads while keeping the `/api/v1/providers/health` response shape
stable and debuggable.

## Problem
`/api/v1/providers/health` can surface error strings from multiple sources:
- Provider plugin bootstrap errors (`plugin_diagnostics.errors_sample[].error`)
- Provider runtime errors (`results[].error`)
- Provider status snapshot errors (`results[].snapshot.last_error`)

These strings may include newlines, excessive whitespace, or very large
messages (e.g. exception chains), which is undesirable for:
- log safety (accidental secret leakage)
- UI rendering stability
- API contract stability (bounded payload expectations)

## Change
- Introduced `_sanitize_error_text()` in `src/api/v1/health.py`:
  - collapses whitespace into a single line
  - truncates to a bounded length (default: 300 chars)
- Applied sanitization to:
  - `plugin_diagnostics.errors_sample[].error`
  - provider snapshot `last_error`
  - provider health `results[].error` (including init-error path)

## Validation
- `.venv/bin/python -m pytest tests/unit/test_provider_health_endpoint.py -v`
  - Result: `4 passed`
- `make validate-core-fast`
  - Result: passed

## Files Changed
- `src/api/v1/health.py`
- `tests/unit/test_provider_health_endpoint.py`

