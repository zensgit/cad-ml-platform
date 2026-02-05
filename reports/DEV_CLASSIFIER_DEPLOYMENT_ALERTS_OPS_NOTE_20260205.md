# DEV_CLASSIFIER_DEPLOYMENT_ALERTS_OPS_NOTE_20260205

## Summary
- Enabled Prometheus alerting rules in the deployment config and added classifier alerts to the bundled rules file.
- Added ops guidance for classifier cache health checks and rate-limit configuration validation.
- Documented the simplified Grafana dashboard limitations and the variable-enabled alternative.

## Changes
- `config/prometheus.yml`: enable `/etc/prometheus/alerting_rules.yml`.
- `config/prometheus/alerting_rules.yml`: add classifier cache hit-rate and rate-limit alerts.
- `docker-compose.observability.yml`: mount `alerting_rules.yml` for Prometheus.
- `docker-compose.observability.yml`: remove Redis host port binding to avoid 6379 conflicts (Redis stays internal).
- `docker-compose.yml`: mount recording + alerting rules for Prometheus.
- `docs/OPERATIONS_MANUAL.md`: troubleshooting steps for classifier cache/rate limiting, including `/health` config checks.
- `docs/OBSERVABILITY_QUICKSTART.md`: note on dashboard variants and variable support.
- `.dockerignore`: exclude `data/training_merged` from Docker build context to reduce disk usage.

## Validation
- `make prom-validate` (recording rules + promtool validation; existing prefix warnings only).
- `docker-compose -f docker-compose.observability.yml up -d --no-deps --no-build prometheus`
- `curl -X POST http://localhost:9090/-/reload`
- `python3 - <<'PY' ...` (queried `http://localhost:9090/api/v1/rules` and confirmed classifier alerts present).
- `python3 - <<'PY' ...` (loaded `config/prometheus.yml` and `config/prometheus/alerting_rules.yml` with `yaml.safe_load`).
- `docker-compose -f docker-compose.observability.yml config | rg -n "redis:" -A6` (confirmed Redis has no host port binding).
- Full stack check (observability):
  - Earlier run (before removing Redis port binding) succeeded with a manual Redis container on the observability network; Prometheus rules and Grafana dashboard were confirmed.
  - After removing the Redis host port binding, the next `docker-compose -f docker-compose.observability.yml up -d --build` attempt failed due to disk exhaustion while copying training data:
    - `COPY data/ ./data/` failed with `input/output error` for `data/training_merged/...`.
    - `df -h .` showed only ~271 MiB free at the time of failure.
  - Docker daemon later unavailable (`docker info` reports missing socket); `open -a Docker` returned error `-1712`, so full-stack restart could not be completed.
  - Added `.dockerignore` to exclude `data/training_merged` so rebuilds no longer copy the large dataset.
  - Follow-up: once Docker Desktop is running and disk space is available, rerun `docker-compose ... up -d --build`.
