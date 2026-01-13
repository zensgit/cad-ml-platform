# GHCR_MULTIARCH_PUBLISH_DESIGN

## Goal
- Publish GHCR images for both linux/amd64 and linux/arm64 so preprod runs natively on arm64 hosts.

## Changes
- Add QEMU setup for multi-arch builds in the GHCR publish workflow.
- Build and push a multi-arch manifest for the main image tags.
- Allow compose users to pin the target platform via `CAD_ML_PLATFORM`.

## Approach
- Use `docker/setup-qemu-action` with Buildx to enable cross-arch builds.
- Set `platforms` to `linux/amd64,linux/arm64` and keep `INSTALL_L3_DEPS` off by default.
- Document platform override usage for GHCR compose deployments.
