# DEV_GHCR_PREPROD_SMOKE_20260113

## Commands
- CAD_ML_IMAGE=ghcr.io/zensgit/cad-ml-platform:main docker compose -f deployments/docker/docker-compose.yml -f deployments/docker/docker-compose.ghcr.yml -f deployments/docker/docker-compose.external-network.yml config > /tmp/cadml-ghcr-smoke-compose.yml
- COMPOSE_FILE=/tmp/cadml-ghcr-smoke-compose.yml CAD_ML_IMAGE=ghcr.io/zensgit/cad-ml-platform:main SKIP_BUILD=1 bash scripts/ci/docker_staging_smoke.sh

## Results
- docker staging smoke completed successfully (health, readiness, cache tuning, metrics).
- /health status: healthy; metrics_enabled: true
- cache_tuning response contained: recommended_capacity, recommended_ttl, confidence
- metrics scrape succeeded (see artifacts)

## Artifacts
- artifacts/docker-staging/health.json
- artifacts/docker-staging/cache_tuning.json
- artifacts/docker-staging/metrics.txt
- artifacts/docker-staging/compose.log

## Notes
- A transient connection reset occurred during initial /health polling, then recovered.
- The smoke script tears down containers and volumes on exit.
