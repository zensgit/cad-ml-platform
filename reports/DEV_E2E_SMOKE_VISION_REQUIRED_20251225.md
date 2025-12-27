# DEV_E2E_SMOKE_VISION_REQUIRED_20251225

## Scope

- Bring up dedupcad-vision locally and rerun E2E smoke with vision required.
- Ensure cad-ml-platform routes can reach dedupcad-vision.

## Changes / Actions

- Started `dedupcad-vision:local` container on `localhost:58001` (mapped to container `8000`).
- Recreated `cad-ml-api` and `cad-ml-dedup2d-worker` with `DEDUPCAD_VISION_URL=http://host.docker.internal:58001`.

## Validation

- Command: `DEDUPCAD_VISION_REQUIRED=1 make e2e-smoke`
- Result: 4 passed

## Notes

- `make` printed warnings about duplicate `security-audit` target (pre-existing).
- `docker-compose` warned that `version` is obsolete and noted orphan containers (pre-existing).
