# Docker Compose Down (2026-01-01)

## Scope

- Stop local cad-ml compose services after validation runs.

## Command

- `docker compose -f deployments/docker/docker-compose.yml -f deployments/docker/docker-compose.minio.yml down`

## Results

- OK: cad-ml api/worker/grafana/prometheus/redis/minio containers stopped and removed.

## Notes

- Docker warned about deprecated `version` fields in compose files.
