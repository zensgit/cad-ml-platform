# DEV_DEDUP2D_STAGING_COMPOSE_OVERRIDE_20251224

## Scope
- Add a staging compose override for Dedup2D (S3 + callback allowlist/HMAC).
- Document local staging usage and callback smoke steps in the runbook.

## Changes
- Added `deployments/docker/docker-compose.dedup2d-staging.yml`.
- Updated `docs/DEDUP2D_STAGING_RUNBOOK.md` with compose usage + callback smoke checks.

## Validation
- Command: `docker compose -f deployments/docker/docker-compose.yml -f deployments/docker/docker-compose.minio.yml -f deployments/docker/docker-compose.dedup2d-staging.yml config`
  - Result: success (warnings about obsolete `version` fields in compose files).
