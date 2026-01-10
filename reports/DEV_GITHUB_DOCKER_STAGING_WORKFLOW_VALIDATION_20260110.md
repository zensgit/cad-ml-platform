# DEV_GITHUB_DOCKER_STAGING_WORKFLOW_VALIDATION_20260110

## Scope
Validate the GitHub Docker staging workflow smoke script against the local Docker Compose stack.

## Attempt 1 (with image rebuild)
- Command: `bash scripts/ci/docker_staging_smoke.sh`
- Result: Blocked while pulling `python:3.9-slim` metadata from Docker Hub (build did not complete after ~110s; aborted).

## Attempt 2 (using existing image)
- Command: `SKIP_BUILD=1 bash scripts/ci/docker_staging_smoke.sh`
- Result:
  - `/health` returned 200 with `metrics_enabled=true`.
  - `POST /api/v1/features/cache/tuning` returned 405, indicating the running `cad-ml-platform:latest` image does not include the latest POST route.
  - Metrics checks did not run because the POST failed.

## Artifacts
- `artifacts/docker-staging/health.json`
- `artifacts/docker-staging/compose.log`

## Follow-up
- Rebuild the image once Docker Hub access is available (`docker compose -f deployments/docker/docker-compose.yml build`).
- Re-run `scripts/ci/docker_staging_smoke.sh` to capture full metrics validation.
- Alternatively, run the GitHub Actions workflow `.github/workflows/docker-staging-smoke.yml` to build in CI.
