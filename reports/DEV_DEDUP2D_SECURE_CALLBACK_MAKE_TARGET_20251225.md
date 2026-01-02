# DEV_DEDUP2D_SECURE_CALLBACK_MAKE_TARGET_20251225

## Scope

- Add a Makefile target for the secure Dedup2D callback smoke test.
- Execute the target to verify it runs end-to-end.

## Changes

- Added `dedup2d-secure-smoke` target in `Makefile`.
- Fixed script temp path handling in `scripts/e2e_dedup2d_secure_callback.sh`.

## Validation

- Command: `DEDUPCAD_VISION_START=1 make dedup2d-secure-smoke`
- Result: succeeded
- Artifact directory: `/tmp/dedup2d_secure_callback.71yCOt`

## Notes

- `make` printed warnings about duplicate `security-audit` target (pre-existing).
- Curl transient connection resets appeared during health retries but the script
  completed successfully.
