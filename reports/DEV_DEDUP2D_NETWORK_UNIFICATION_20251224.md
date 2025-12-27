# DEV_DEDUP2D_NETWORK_UNIFICATION_20251224

## Scope
- Pin the Docker network name to avoid per-project prefixed networks.
- Document network recreation guidance in staging runbook.

## Changes
- `deployments/docker/docker-compose.yml`: set `networks.cad-ml-network.name=cad-ml-network`.
- `docs/DEDUP2D_STAGING_RUNBOOK.md`: add note about shared network name and recreating old containers.

## Validation
- Command: `docker compose -f deployments/docker/docker-compose.yml -f deployments/docker/docker-compose.minio.yml -f deployments/docker/docker-compose.dedup2d-staging.yml config`
  - Result: success (warnings about obsolete `version` fields in compose files).
