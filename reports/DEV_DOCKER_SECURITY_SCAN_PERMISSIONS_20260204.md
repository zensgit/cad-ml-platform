# DEV_DOCKER_SECURITY_SCAN_PERMISSIONS_20260204

## Summary
Granted `security-events: write` permission to the multi-arch Docker workflow so the Trivy SARIF
upload step can publish results without failing with "Resource not accessible by integration".

## Changes
- Added `security-events: write` to the workflow permissions in
  `.github/workflows/docker-multiarch.yml`.

## Validation
- CI will re-run the Multi-Architecture Docker Build workflow to confirm the
  Security Scan completes successfully.
