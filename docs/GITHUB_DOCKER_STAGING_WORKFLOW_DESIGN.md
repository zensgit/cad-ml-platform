# GitHub Docker Staging Workflow Design

## Scope
Provide a GitHub Actions workflow and local smoke script that simulate a staging deployment using Docker Compose when no dedicated staging environment is available.

## Workflow Overview
- Workflow: `.github/workflows/docker-staging-smoke.yml`
- Script: `scripts/ci/docker_staging_smoke.sh`
- Compose file: `deployments/docker/docker-compose.yml`

## Smoke Checks
- `/health` returns 200 and reports `metrics_enabled=true`.
- `/ready` returns 200 to confirm readiness before smoke requests.
- `POST /api/v1/features/cache/tuning` returns a recommendation payload.
- `/metrics` exposes cache tuning metrics:
  - `feature_cache_tuning_requests_total`
  - `feature_cache_tuning_recommended_capacity`
  - `feature_cache_tuning_recommended_ttl_seconds`
- Grafana dashboard JSON validated via `python3 scripts/validate_dashboard_metrics.py`.

## Inputs and Environment
- `api_port` (workflow input, default `8000`) controls the exposed API port.
- `CAD_ML_API_PORT` and `API_PORT` are passed to the script.
- `API_KEY` defaults to `test` for the X-API-Key header.
- `SKIP_BUILD=1` can be set when a local `cad-ml-platform:latest` image already exists to avoid rebuilding.
- `INSTALL_L3_DEPS=0` skips heavy L3 dependencies (e.g., `pythonocc-core`) for staging smoke builds.

## Artifacts
- `artifacts/docker-staging/health.json`
- `artifacts/docker-staging/cache_tuning.json`
- `artifacts/docker-staging/metrics.txt`
- `artifacts/docker-staging/compose.log`

## Failure Modes
- Docker Hub throttling can block image builds; rerun with cached layers or pre-pull the base image.
- If `POST /api/v1/features/cache/tuning` returns 405, rebuild the image to ensure the latest API routes are present.
