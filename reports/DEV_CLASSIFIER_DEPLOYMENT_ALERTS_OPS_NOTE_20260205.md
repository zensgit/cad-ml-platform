# DEV_CLASSIFIER_DEPLOYMENT_ALERTS_OPS_NOTE_20260205

## Summary
- Enabled Prometheus alerting rules in the deployment config and added classifier alerts to the bundled rules file.
- Added ops guidance for classifier cache health checks and rate-limit configuration validation.
- Documented the simplified Grafana dashboard limitations and the variable-enabled alternative.

## Changes
- `config/prometheus.yml`: enable `/etc/prometheus/alerting_rules.yml`.
- `config/prometheus/alerting_rules.yml`: add classifier cache hit-rate and rate-limit alerts.
- `docker-compose.observability.yml`: mount `alerting_rules.yml` for Prometheus.
- `docker-compose.yml`: mount recording + alerting rules for Prometheus.
- `docs/OPERATIONS_MANUAL.md`: troubleshooting steps for classifier cache/rate limiting, including `/health` config checks.
- `docs/OBSERVABILITY_QUICKSTART.md`: note on dashboard variants and variable support.

## Validation
- `make prom-validate` (recording rules + promtool validation; existing prefix warnings only).
- `docker-compose -f docker-compose.observability.yml up -d --no-deps --no-build prometheus`
- `curl -X POST http://localhost:9090/-/reload`
- `python3 - <<'PY' ...` (queried `http://localhost:9090/api/v1/rules` and confirmed classifier alerts present).
- `python3 - <<'PY' ...` (loaded `config/prometheus.yml` and `config/prometheus/alerting_rules.yml` with `yaml.safe_load`).
- Full stack check (observability):
  - `docker-compose -f docker-compose.observability.yml up -d --build`
  - Port conflict on `6379` (existing Redis); started Redis on the observability network without host binding:
    - `docker run -d --name cad-redis --network cad-observability --network-alias redis redis:6-alpine`
    - `docker-compose -f docker-compose.observability.yml up -d --no-deps --no-build app grafana prometheus`
  - `http://localhost:9090/api/v1/targets` shows `cad-ml-platform`, `prometheus`, `redis`, `node` targets registered.
  - `http://localhost:9090/api/v1/rules` lists `ClassifierCacheHitRateLow` and `ClassifierRateLimitedHigh`.
  - Grafana API search confirmed `CAD ML Platform - Observability (Enhanced)` dashboard is provisioned.
