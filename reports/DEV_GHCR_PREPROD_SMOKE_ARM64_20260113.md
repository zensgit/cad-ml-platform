# DEV_GHCR_PREPROD_SMOKE_ARM64_20260113

## Scope
Run the docker staging smoke script against the GHCR image using the linux/arm64 platform override.

## Commands
- `CAD_ML_IMAGE=ghcr.io/zensgit/cad-ml-platform:main CAD_ML_PLATFORM=linux/arm64 docker compose -f deployments/docker/docker-compose.yml -f deployments/docker/docker-compose.ghcr.yml -f deployments/docker/docker-compose.external-network.yml config > /tmp/cadml-ghcr-smoke-arm64-compose.yml`
- `COMPOSE_FILE=/tmp/cadml-ghcr-smoke-arm64-compose.yml CAD_ML_IMAGE=ghcr.io/zensgit/cad-ml-platform:main CAD_ML_PLATFORM=linux/arm64 SKIP_BUILD=1 ARTIFACT_DIR=artifacts/docker-staging-arm64-20260113 bash scripts/ci/docker_staging_smoke.sh`

## Results
- Smoke checks completed successfully on linux/arm64.
- `/health` returned `metrics_enabled=true`.
- Cache tuning endpoint returned the expected recommendation fields.
- Metrics endpoint exposed cache tuning metrics.

## Notes
- One transient `curl` connection reset occurred during the initial health wait; retries succeeded.
- Docker reported an orphan container warning for `metasheet-dev-postgres` (pre-existing, not part of this compose run).

## Artifacts
- `artifacts/docker-staging-arm64-20260113/health.json`
- `artifacts/docker-staging-arm64-20260113/cache_tuning.json`
- `artifacts/docker-staging-arm64-20260113/metrics.txt`
- `artifacts/docker-staging-arm64-20260113/compose.log`
