# Prometheus → Alertmanager Alert Chain Verification

- Date: 2025-12-27
- Scope: Validate Prometheus alert routing to Alertmanager using a synthetic rule

## Steps
- Added Alertmanager target to `deployments/docker/prometheus.yml`
- Started Alertmanager service via docker compose
- Added temporary rule `prometheus/alerts/test_alert.yml` (CadMlTestAlert)
- Restarted Prometheus to load rule
- Verified alert firing in Prometheus and received by Alertmanager
- Removed temporary rule and restarted Prometheus

## Commands
- docker compose -f deployments/docker/docker-compose.yml up -d alertmanager
- docker restart cad-ml-prometheus
- python3 - <<'PY' ... /api/v1/rules (CadMlTestAlert firing)
- python3 - <<'PY' ... /api/v2/alerts (CadMlTestAlert active)
- rm prometheus/alerts/test_alert.yml
- docker restart cad-ml-prometheus

## Result
- PASS

## Notes
- Alertmanager v2 API used for verification
- Temporary rule removed after validation
