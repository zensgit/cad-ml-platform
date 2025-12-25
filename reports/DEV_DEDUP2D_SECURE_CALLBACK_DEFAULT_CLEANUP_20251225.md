# DEV_DEDUP2D_SECURE_CALLBACK_DEFAULT_CLEANUP_20251225

## Scope

- Enable default cleanup for `make dedup2d-secure-smoke`.
- Validate the target still runs end-to-end.

## Changes

- Updated `Makefile` to default `DEDUP2D_SECURE_SMOKE_CLEANUP=1` for `dedup2d-secure-smoke`.

## Validation

- Command: `DEDUPCAD_VISION_START=1 make dedup2d-secure-smoke`
- Result: succeeded
- Cleanup check: `/tmp/dedup2d_secure_callback.gX4w4C` removed after completion

## Notes

- `make` printed warnings about duplicate `security-audit` target (pre-existing).
- Curl transient connection resets appeared during health retries but the script
  completed successfully.
