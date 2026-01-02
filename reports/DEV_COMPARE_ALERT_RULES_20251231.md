# Compare Alert Rules (2025-12-31)

## Scope

- Add Prometheus alert rules for `/api/compare` failure rates.

## Changes

- `config/prometheus/alerting_rules.yml`
  - `CompareRequestFailureRateHigh`
  - `CompareNotFoundDominant`

## Validation

```bash
docker run --rm --entrypoint promtool \
  -v "$(pwd)":/workspace:ro \
  prom/prometheus:latest \
  check rules /workspace/config/prometheus/alerting_rules.yml
```

Result:
- `SUCCESS: 49 rules found`

Note: `scripts/validate_prometheus.sh` uses Docker without `--entrypoint`, which fails in this environment; direct `promtool` invocation via Docker succeeded.
