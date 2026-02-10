# DEV_PROVIDER_CHECK_METRICS_EXPOSITION_TEST_20260210

## Summary
- Added unit coverage to ensure the new core-provider check metrics are actually emitted in `/metrics`
  after running provider health/readiness checks.

## What’s Covered
- `core_provider_checks_total{source,domain,provider,result}`
- `core_provider_check_duration_seconds{source,domain,provider}`

## Implementation
- `tests/unit/test_provider_check_metrics_exposed.py`
  - Calls `/api/v1/providers/health` with a patched provider registry to force a deterministic provider check.
  - Calls `check_provider_readiness()` with an unregistered provider to force the `init_error` path.
  - Scrapes `/metrics` and asserts the labeled samples are present (order-independent).

## Validation
- `pytest -q tests/unit/test_provider_check_metrics_exposed.py`
  - Result: pass

