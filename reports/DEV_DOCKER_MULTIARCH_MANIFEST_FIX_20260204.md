# DEV_DOCKER_MULTIARCH_MANIFEST_FIX_20260204

## Summary
- Fixed multi-arch manifest creation to use metadata JSON output instead of raw tag list.
- Resolves `jq` parse errors and downstream Trivy scan failures caused by missing manifests.

## Change
- `.github/workflows/docker-multiarch.yml`
  - Added `json` output from docker/metadata-action.
  - Updated `DOCKER_METADATA_OUTPUT_JSON` to use `needs.metadata.outputs.json`.

## Verification
- CI Multi-Architecture Docker Build and Security Scan should pass once the manifest is created correctly.
