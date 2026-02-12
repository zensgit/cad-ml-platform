# DEV_CORE_PROVIDER_PLUGIN_ERRORS_SANITIZATION_20260212

## Goal
Ensure core provider plugin error strings surfaced via health endpoints are:
- single-line (no embedded newlines / whitespace explosions)
- bounded in size (avoid leaking large exception context)

This applies both to:
- `/health` payload (`config.core_providers.plugins.errors`)
- `/api/v1/providers/registry` (`registry.plugins.errors`)

## Change
- Added shared utility: `src/utils/text_sanitize.py`
  - `sanitize_single_line_text(value, max_len=300)` collapses whitespace and truncates.
- Applied sanitization in:
  - `src/api/health_utils.py` when attaching `config.core_providers`
  - `src/api/v1/health.py` for `/api/v1/providers/registry`
- Refactored `/api/v1/providers/health` sanitization to reuse the shared helper
  (behavior unchanged, but now consistent across endpoints).

## Validation
- `.venv/bin/python -m pytest tests/unit/test_provider_health_endpoint.py tests/unit/test_health_utils_coverage.py -v`
  - Result: `24 passed`
- `make validate-core-fast`
  - Result: passed

## Files Changed
- `src/utils/text_sanitize.py`
- `src/api/health_utils.py`
- `src/api/v1/health.py`
- `tests/unit/test_health_utils_coverage.py`
- `tests/unit/test_provider_health_endpoint.py`

