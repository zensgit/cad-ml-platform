# GHCR Multi-Arch Image Refresh (Week 7)

## Summary
- Built and pushed a multi-arch `dedupcad-vision` image (amd64 + arm64).
- Updated CI default digest and docs to the new manifest list.

## Image Details
- Tag: `ghcr.io/zensgit/dedupcad-vision:1.1.1`
- Digest: `ghcr.io/zensgit/dedupcad-vision@sha256:9f7f567e3b0c1c882f9a363f1b1cb095d30d9e9b184e582d6b19ec7446a86251`
- Platforms: `linux/amd64`, `linux/arm64`

## Code Changes
- `.github/workflows/ci.yml` (DEDUPCAD_VISION_IMAGE default digest)
- `README.md` (override example)
- `docs/CI_OPTIMIZATION_SUMMARY.md` (override example)

## Verification
- `docker manifest inspect ghcr.io/zensgit/dedupcad-vision@sha256:9f7f...`
  - Confirmed amd64 + arm64 manifests.
- `docker pull --platform linux/amd64 ghcr.io/zensgit/dedupcad-vision@sha256:9f7f...`
  - Result: PASS
- `docker run ... ghcr.io/zensgit/dedupcad-vision@sha256:9f7f...` + `curl /health`
  - Result: PASS (`dedupcad-vision ready`)

## Notes
- This resolves the CI e2e-smoke failure caused by the previous arm64-only digest.
