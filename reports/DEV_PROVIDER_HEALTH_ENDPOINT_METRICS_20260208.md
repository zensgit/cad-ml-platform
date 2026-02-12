# DEV_PROVIDER_HEALTH_ENDPOINT_METRICS_20260208

## Summary

Instrumented the provider registry + provider health visibility endpoints with the existing health Prometheus metrics (`health_requests_total`, `health_request_duration_seconds`) via `record_health_request`.

## Changes

- `src/api/v1/health.py`
  - Added `record_health_request(...)` for:
    - `GET /api/v1/providers/registry` (`endpoint=providers_registry`)
    - `GET /api/v1/providers/health` (`endpoint=providers_health`)
  - Metrics are best-effort and never block responses.

## Validation

Executed:

```bash
python3 -m pytest -q tests/unit/test_provider_health_endpoint.py
```

Result: PASS

