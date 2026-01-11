# Health Endpoint Metrics Design

## Scope
- Add Prometheus metrics for `/health`, `/health/extended`, and `/ready`.
- Ensure metrics contract tests validate the new health metrics.
- Document the new metrics in the metrics index.

## Problem Statement
- Health endpoints had no request counters or latency tracking.
- Metrics contract tests did not cover health endpoint observability.

## Design
- Add `health_requests_total{endpoint,status}` and `health_request_duration_seconds{endpoint}`
  in `src/utils/metrics.py`.
- Record metrics in `/health`, `/health/extended`, and `/ready` via `record_health_request`.
- Update metrics contract tests to register and validate the new metrics.
- Document new metrics in `docs/METRICS_INDEX.md`.

## Impact
- Operators can track health endpoint usage and latency trends.
- Metrics contract suite enforces health metric availability.

## Validation
- `python3 -m pytest tests/test_metrics_contract.py -k "required_metrics_present or metric_label_schemas or histogram_metrics_have_buckets" -v`
