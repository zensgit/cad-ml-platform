# Maintenance: Compose Version Removal & Prometheus URL (2026-01-01)

## Scope

- Remove deprecated `version` fields from docker compose files.
- Allow Prometheus URL override for cardinality audit targets (default 9091).

## Changes

- `deployments/docker/docker-compose.yml`: removed `version` field.
- `deployments/docker/docker-compose.minio.yml`: removed `version` field.
- `Makefile`: added `PROMETHEUS_URL` (default `http://localhost:9091`) and wired it into `metrics-audit`, `cardinality-check`, and `metrics-audit-watch`.

## Tests

- Not run (config-only change).
