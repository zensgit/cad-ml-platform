# Alertmanager Test Alert Verification

- Date: 2025-12-27
- Scope: Trigger a test alert and verify it appears in Alertmanager

## Commands
- docker run -d --rm --name cad-alertmanager -p 9093:9093 -v config/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro prom/alertmanager --config.file=/etc/alertmanager/alertmanager.yml
- curl -XPOST http://localhost:9093/api/v2/alerts (TestAlert payload)
- python3 - <<'PY' ... fetch /api/v2/alerts and verify TestAlert state
- docker stop cad-alertmanager

## Result
- PASS

## Notes
- Alertmanager v2 API required for posting alerts
- TestAlert observed in active state via /api/v2/alerts
