# DEV_DEDUP2D_RUNBOOK_VISION_URL_20251225

## Scope

- Document DEDUPCAD_VISION_URL usage when DedupCAD Vision runs in the same docker network.
- Clean up local test artifacts and stop the temporary Vision container.

## Changes

- Updated `docs/DEDUP2D_STAGING_RUNBOOK.md` with the `DEDUPCAD_VISION_URL` note for
  containerized Vision.
- Removed temporary smoke-test artifacts under `/tmp` and stopped `dedupcad-vision-api`.

## Validation

- Not run (documentation + cleanup only).

## Notes

- The local image `dedupcad-vision:local` remains available for future runs.
