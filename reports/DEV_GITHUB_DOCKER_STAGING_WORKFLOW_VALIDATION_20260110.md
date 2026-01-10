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

## Attempt 3 (GitHub Actions)
- Workflow: `.github/workflows/docker-staging-smoke.yml` on `main`
- Result: Failed during Docker build because `filelock==3.20.1` was not available on PyPI for the Python 3.9 base image.
- Follow-up: Pin `filelock` to `3.19.1` in `requirements.txt` and re-run the workflow.

## Attempt 4 (GitHub Actions after filelock fix)
- Workflow: `.github/workflows/docker-staging-smoke.yml` on `main`
- Result: Docker build failed on dependency resolution: `urllib3==2.6.0` conflicts with `botocore` constraints.
- Follow-up: Pin `urllib3` to `2.1.0` and re-run the workflow.

## Attempt 5 (GitHub Actions after urllib3 pin)
- Workflow: `.github/workflows/docker-staging-smoke.yml` on `main`
- Result: Docker build still failed on dependency resolution with `urllib3==2.1.0`.
- Follow-up: Pin `urllib3` to `2.0.7` and re-run the workflow.

## Attempt 6 (GitHub Actions after urllib3 2.0.7)
- Workflow: `.github/workflows/docker-staging-smoke.yml` on `main`
- Result: Docker build still failed on dependency resolution with `urllib3==2.0.7`.
- Follow-up: Pin `urllib3` to `1.26.20` and re-run the workflow.

## Attempt 7 (GitHub Actions after urllib3 1.26.20)
- Workflow: `.github/workflows/docker-staging-smoke.yml` on `main`
- Result: Docker build failed because `pythonocc-core>=7.7.0` has no matching wheel for Python 3.9.
- Follow-up: Update the Docker base image to `python:3.10-slim` and re-run the workflow.

## Attempt 8 (GitHub Actions after Python 3.10 base)
- Workflow: `.github/workflows/docker-staging-smoke.yml` on `main`
- Result: Docker build still failed on `pythonocc-core>=7.7.0` wheel availability.
- Follow-up: Add `INSTALL_L3_DEPS=0` to the staging workflow to skip L3-only dependencies.

## Attempt 9 (GitHub Actions with L3 deps skipped)
- Workflow: `.github/workflows/docker-staging-smoke.yml` on `main`
- Result: Stack started, but curl connection resets during `/health` and missing cache tuning metrics caused the smoke check to exit.
- Follow-up: Add retry logic for POST/metrics and require cache tuning metrics with backoff.

## Attempt 10 (GitHub Actions with retries)
- Workflow: `.github/workflows/docker-staging-smoke.yml` on `main`
- Result: Cache tuning metrics still missing after retries; added metrics snapshot logging for debugging.

## Attempt 11 (GitHub Actions with diagnostics)
- Workflow: `.github/workflows/docker-staging-smoke.yml` on `main`
- Result: Cache tuning metrics still missing after retries; added readiness wait and absolute artifact paths, and disabled DEBUG in compose.

## Attempt 12 (GitHub Actions metrics redirect)
- Workflow: `.github/workflows/docker-staging-smoke.yml` on `main`
- Result: `/metrics` returned 307 redirect; metrics file empty. Updated smoke script to request `/metrics/` and follow redirects.

## Attempt 13 (GitHub Actions success)
- Workflow: `.github/workflows/docker-staging-smoke.yml` on `main`
- Result: Workflow succeeded; smoke and dashboard metrics validations passed.

## Attempt 14 (Local Docker Compose)
- Command: `bash scripts/ci/docker_staging_smoke.sh`
- Result: Local build and compose smoke checks passed (health, readiness, cache tuning, metrics).

## Artifacts
- `artifacts/docker-staging/health.json`
- `artifacts/docker-staging/compose.log`

## Follow-up
- GitHub Actions staging smoke now passes after the `/metrics/` redirect fix.
- Rebuild the image once Docker Hub access is available (`docker compose -f deployments/docker/docker-compose.yml build`) for local validation as needed.
