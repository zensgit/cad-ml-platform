# DEV_PROVIDER_CHECK_METRICS_20260210

## Summary
- Added Prometheus metrics for core provider framework readiness/health checks.
- Instrumented both readiness checks (`/ready`) and provider health endpoints (`/api/v1/providers/health`) with per-provider counters and latency histograms.

## Metrics
- `core_provider_checks_total{source,domain,provider,result}`
  - `source`: `readiness` | `providers_health`
  - `result`: `ready` | `down` | `init_error`
- `core_provider_check_duration_seconds{source,domain,provider}`

## Implementation
- `src/utils/metrics.py`
  - Added the metric definitions.
- `src/core/providers/readiness.py`
  - Records per-provider metrics during readiness probes.
- `src/api/v1/health.py`
  - Records per-provider metrics for `/api/v1/providers/health`.

## Validation
- `pytest -q tests/unit/test_provider_health_endpoint.py tests/unit/test_provider_readiness.py tests/unit/test_readiness_coverage.py`
  - Result: pass

## Notes
- Metrics are best-effort and should never break readiness/health responses.

