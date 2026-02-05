# DEV_CLASSIFIER_DASHBOARD_ALERTS_HEALTH_20260205

## Summary
Extended classifier observability with dashboard panels, alert rules, and a
health endpoint for cache inspection.

## Changes
- Added classifier cache panels to Grafana dashboards:
  - `grafana/dashboards/observability.json`
  - `docs/grafana/observability_dashboard.json`
- Added classifier alert rules: `prometheus/alerts/classifier.yml`.
- Added classifier cache stats endpoint:
  - `GET /api/v1/health/classifier/cache` (admin token required)
  - Alias `GET /api/v1/classifier/cache`
- Added health endpoint documentation and alert rule references:
  - `docs/HEALTH_ENDPOINT_CONFIG.md`
  - `docs/ALERT_RULES.md`
- Added unit test: `tests/unit/test_health_classifier_cache.py`.

## Validation
- `python3 -m pytest tests/unit/test_health_classifier_cache.py -q`
- `python3 scripts/validate_dashboard_metrics.py`

## Notes
- Alert thresholds are conservative defaults; tune based on production traffic.
