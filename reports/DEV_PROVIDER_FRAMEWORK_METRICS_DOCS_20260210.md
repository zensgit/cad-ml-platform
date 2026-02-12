# DEV_PROVIDER_FRAMEWORK_METRICS_DOCS_20260210

## Summary
- Documented core provider framework Prometheus metrics in the provider framework guide.

## Changes
- `docs/PROVIDER_FRAMEWORK.md`
  - Added an "Observability (Prometheus Metrics)" section:
    - `core_provider_checks_total{source,domain,provider,result}`
    - `core_provider_check_duration_seconds{source,domain,provider}`

## Validation
- `pytest -q tests/unit/test_provider_check_metrics_exposed.py`
  - Result: pass

