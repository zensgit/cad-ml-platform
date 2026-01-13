# DEV_GHCR_MULTIARCH_PUBLISH_VALIDATION_20260113

## Scope
Validate that the GHCR publish workflow completed successfully and produced a multi-arch manifest with linux/amd64 and linux/arm64 images, and that the arm64 variant can be pulled.

## Workflow Run
- Run: `gh run watch 20959475194 --exit-status`
- Result: Success (build-and-push job completed)

## Manifest Inspection
Command:
```
docker buildx imagetools inspect ghcr.io/zensgit/cad-ml-platform:main
```
Result:
- Index digest: `sha256:a329a073023704a4d5a314f4ecf690a494e4e03e82507fb92a3720097b477045`
- Platforms:
  - `linux/amd64`
  - `linux/arm64`
- Attestation manifests present (`unknown/unknown` with `vnd.docker.reference.type: attestation-manifest`).

## Arm64 Pull Validation
Command:
```
docker pull --platform linux/arm64 ghcr.io/zensgit/cad-ml-platform:main
```
Result:
- Digest: `sha256:a329a073023704a4d5a314f4ecf690a494e4e03e82507fb92a3720097b477045`
- Status: Downloaded newer image for `ghcr.io/zensgit/cad-ml-platform:main`

## Notes
- The manifest list includes both target architectures, confirming multi-arch publish succeeded.
