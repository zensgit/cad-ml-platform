# GHCR_PREPROD_SMOKE_VALIDATION_DESIGN

## Goal
- Validate the GHCR-based preprod path using the existing docker staging smoke script.

## Approach
- Generate a merged compose file from base + GHCR + external network overrides.
- Run `scripts/ci/docker_staging_smoke.sh` with `SKIP_BUILD=1` to avoid local builds.
- Capture health, readiness, cache tuning, and metrics artifacts for verification.
