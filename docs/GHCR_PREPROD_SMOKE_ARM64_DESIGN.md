# GHCR_PREPROD_SMOKE_ARM64_DESIGN

## Goal
- Validate GHCR preprod smoke coverage against linux/arm64 images.

## Approach
- Generate a merged compose file with `CAD_ML_PLATFORM=linux/arm64` and the GHCR image override.
- Run `scripts/ci/docker_staging_smoke.sh` with `SKIP_BUILD=1` and a dedicated artifacts directory.
- Verify health, readiness, cache tuning response, and metrics exposure on the arm64 container.
