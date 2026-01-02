# Metrics Pushgateway Report

- Date: 2025-12-27
- Scope: Push metrics to a live Pushgateway via `PUSHGATEWAY_URL`

## Setup
- Started Pushgateway container: `docker run -d --rm --name cad-pushgateway -p 9092:9091 prom/pushgateway`
- Note: host port 9091 already mapped to Prometheus; used 9092 for Pushgateway

## Command
- PUSHGATEWAY_URL=http://localhost:9092 make metrics-push
- curl -s http://localhost:9092/metrics | rg -n "cad_ml_evaluation"

## Result
- PASS

## Notes
- Fixed Prometheus output to be Pushgateway-compatible:
  - `cad_ml_evaluation_info` type changed to `gauge`
  - Added trailing newline to exposition output
- Verified `cad_ml_evaluation_*` metrics present in Pushgateway output
