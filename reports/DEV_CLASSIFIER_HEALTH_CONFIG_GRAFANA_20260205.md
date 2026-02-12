# DEV_CLASSIFIER_HEALTH_CONFIG_GRAFANA_20260205

## Summary
Integrated classifier alert rules into Prometheus assets, exposed classifier
cache/rate-limit settings via `/health`, and added Grafana variable support for
classifier panels in the documentation dashboard.

## Changes
- Config:
  - Added `CLASSIFIER_CACHE_MAX_SIZE` to `docs/DEPLOYMENT.md`.
  - Exposed classifier cache/rate-limit settings in `/health` config payload:
    `src/api/health_models.py`, `src/api/health_utils.py`.
  - Wired classifier cache size to env in `src/inference/classifier_api.py`.
- Dashboards:
  - Added `instance` variable and instance-scoped classifier panels to
    `docs/grafana/observability_dashboard.json`.
- Alerts:
  - Prometheus already loads `prometheus/alerts/*.yml`; the classifier alert
    rules added earlier are now covered by deployment config.

## Tests
- `python3 -m pytest tests/unit/test_health_classifier_cache.py -q`

## Notes
- The provisioned Grafana dashboard JSON under `grafana/dashboards/` uses a
  simplified schema, so instance variables were added only to the docs
  dashboard (`docs/grafana/...`).
