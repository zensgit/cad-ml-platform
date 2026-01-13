# GHCR_PREPROD_COMPOSE_DESIGN

## Goal
- Offer a staging-like deployment path without a staging account by using GHCR images and a compose override.

## Changes
- Add a GHCR publish workflow for main branch images.
- Provide a compose override that pulls prebuilt images and disables local builds.
- Document usage in README and a dedicated staging guide.

## Approach
- Publish tags `main` and `sha-<commit>` to GHCR for traceable preprod deployments.
- Use `CAD_ML_IMAGE` to pin the image in `docker-compose.ghcr.yml`.
- Use `--no-build` with the GHCR override to avoid local builds.
- Validate compose config locally before sharing the workflow.
- Default GHCR builds skip `requirements-l3.txt` via `INSTALL_L3_DEPS=0` to avoid optional heavy deps.
- GHCR images publish `linux/amd64` and `linux/arm64`; compose can pin via `CAD_ML_PLATFORM`.
- Provide an external network override when `cad-ml-network` already exists with mismatched labels.
