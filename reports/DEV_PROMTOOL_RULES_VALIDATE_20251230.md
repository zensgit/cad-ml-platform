# Prometheus Rules Validation Report

- Date: 2025-12-30
- Tool: promtool 2.49.1 (via Docker image prom/prometheus:v2.49.1)
- Targets:
  - config/prometheus/recording_rules.yml
  - config/prometheus/alerting_rules.yml

## Commands
```bash
docker run --rm --entrypoint promtool -v "$PWD/config/prometheus:/etc/prometheus" \
  prom/prometheus:v2.49.1 check rules /etc/prometheus/recording_rules.yml

docker run --rm --entrypoint promtool -v "$PWD/config/prometheus:/etc/prometheus" \
  prom/prometheus:v2.49.1 check rules /etc/prometheus/alerting_rules.yml
```

## Results
- recording_rules.yml: SUCCESS (46 rules found)
- alerting_rules.yml: SUCCESS (47 rules found)

## Notes
- Duplicate recording rules were removed prior to validation (cad:memory_fallback_ratio:5m,
  cad:model_security_fail_rate:5m).
