# DEV_DEDUP2D_SECURE_CALLBACK_CLEANUP_20251225

## Scope

- Add optional cleanup for the secure callback smoke script.
- Document the cleanup flag in the staging runbook.
- Validate the updated make target with cleanup enabled.

## Changes

- Added `DEDUP2D_SECURE_SMOKE_CLEANUP` handling to `scripts/e2e_dedup2d_secure_callback.sh`.
- Documented the cleanup flag in `docs/DEDUP2D_STAGING_RUNBOOK.md`.

## Validation

- Command: `DEDUPCAD_VISION_START=1 DEDUP2D_SECURE_SMOKE_CLEANUP=1 make dedup2d-secure-smoke`
- Result: succeeded
- Cleanup check: `/tmp/dedup2d_secure_callback.ne8jp3` removed after completion

## Notes

- `make` printed warnings about duplicate `security-audit` target (pre-existing).
- Curl transient connection resets appeared during health retries but the script
  completed successfully.
