# DEV_DEDUP2D_CALLBACK_HMAC_ALLOWLIST_20251224

## Scope
- Validate Dedup2D async webhook callback with HMAC + allowlist enforced in staging (S3 backend).

## Setup
- Started local callback server on `http://host.docker.internal:57233/hook` (writes to `/tmp/dedup2d_callback.json`).
- Recreated `cad-ml-api` and `dedup2d-worker` with callback env enabled:
  - `DEDUP2D_CALLBACK_ALLOW_HTTP=1`
  - `DEDUP2D_CALLBACK_BLOCK_PRIVATE_NETWORKS=1`
  - `DEDUP2D_CALLBACK_ALLOWLIST=host.docker.internal`
  - `DEDUP2D_CALLBACK_HMAC_SECRET=stage_secret`
  - `DEDUP2D_CALLBACK_TIMEOUT_SECONDS=2`, `DEDUP2D_CALLBACK_MAX_ATTEMPTS=1`
- Started local dedupcad-vision on `http://127.0.0.1:58001`.

## Validation
- Allowlist rejection:
  - Command: `POST /api/v1/dedup/2d/search?...&callback_url=http://example.com/hook`
  - Result: HTTP `400` with `CALLBACK_URL_INVALID` (hostname not in allowlist).
- Callback success:
  - Command: `POST /api/v1/dedup/2d/search?...&callback_url=http://host.docker.internal:57233/hook`
  - Result: job `6e70feae-c36b-441f-a097-b3967c7814a8` completed.
  - Job metadata: `callback_status=success`, `callback_attempts=1`, `callback_http_status=200`.
  - Callback payload received at `/tmp/dedup2d_callback.json`.
  - HMAC signature verified with `stage_secret` (format `t=...,v1=...`).

## Notes
- Callback server and dedupcad-vision were stopped after validation.
