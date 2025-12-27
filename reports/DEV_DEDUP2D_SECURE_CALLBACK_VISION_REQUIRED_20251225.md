# DEV_DEDUP2D_SECURE_CALLBACK_VISION_REQUIRED_20251225

## Scope

- Validate Dedup2D secure callback flow with dedupcad-vision reachable on host.

## Validation

- Command: `DEDUPCAD_VISION_URL=http://host.docker.internal:58001 DEDUPCAD_VISION_START=0 make dedup2d-secure-smoke`
- Result: succeeded
- Cleanup: `/tmp/dedup2d_secure_callback.jlN8gx` removed after completion

## Notes

- `make` printed warnings about duplicate `security-audit` target (pre-existing).
- `docker compose` warned about obsolete `version` and orphan containers (pre-existing).
- Curl transient connection resets appeared during health retries but the script completed.
