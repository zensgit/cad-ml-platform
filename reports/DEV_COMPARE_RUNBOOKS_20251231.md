# Compare Alert Runbooks (2025-12-31)

## Scope

- Add runbooks for compare failure rate and not_found dominance alerts.

## Changes

- `docs/runbooks/compare_failure_rate.md`
- `docs/runbooks/compare_not_found.md`
- `config/prometheus/alerting_rules.yml`: update runbook URLs to local docs.

## Validation

```bash
docker run --rm --entrypoint promtool \
  -v "$(pwd)":/workspace:ro \
  prom/prometheus:latest \
  check rules /workspace/config/prometheus/alerting_rules.yml
```

Result:
- `SUCCESS: 49 rules found`
