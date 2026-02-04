# DEV_DOCKER_SECURITY_SCAN_TAG_FIX_20260204

## Summary
Adjusted the multi-arch Docker workflow so the Security Scan waits for manifest creation and scans a
known published tag from metadata outputs. This avoids Trivy failing to resolve a non-existent
full-SHA tag and ensures the scan runs against a pushed image.

## Changes
- `security-scan` now depends on `metadata` + `merge` so tags exist before scanning.
- `image-ref` uses the metadata version tag instead of `github.sha`.

## Validation
- CI will re-run `Multi-Architecture Docker Build` with the corrected dependency ordering.

## Notes
- Previous failures were caused by Trivy scanning `ghcr.io/zensgit/cad-ml-platform:<full_sha>`
  before the tag existed in the registry.
