# Dedup2D Observability Local Checks (2026-01-01)

## Scope

- Confirm Dedup2D alert rules are loaded in Prometheus and dashboard is present in Grafana.

## Commands

- `curl -s http://localhost:9091/api/v1/rules | rg -n "dedup2d"`
- `curl -s -u admin:admin 'http://localhost:3001/api/search?query=Dedup2D'`

## Results

- Prometheus rules API shows `dedup2d` alert group loaded from `/etc/prometheus/alerts/dedup2d.yml`.
- Grafana search returns `Dedup2D Dashboard` (uid `dedup2d-dashboard`).
