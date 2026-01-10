# DEV_DEDUP2D_CALLBACK_HMAC_AFTER_NETWORK_20251224

## Scope
- Re-run Dedup2D callback/HMAC smoke test after network standardization.

## Setup
- Started local callback server on `http://host.docker.internal:56874/hook`.
- Recreated `cad-ml-api` + `dedup2d-worker` with callback env:
  - `DEDUP2D_CALLBACK_ALLOW_HTTP=1`
  - `DEDUP2D_CALLBACK_BLOCK_PRIVATE_NETWORKS=1`
  - `DEDUP2D_CALLBACK_RESOLVE_DNS=0`
  - `DEDUP2D_CALLBACK_ALLOWLIST=host.docker.internal`
  - `DEDUP2D_CALLBACK_HMAC_SECRET=stage_secret`
- Started local dedupcad-vision on `http://127.0.0.1:58001`.

## Validation
- Allowlist rejection:
  - Command: `POST /api/v1/dedup/2d/search?...&callback_url=http://example.com/hook`
  - Result: HTTP `400` with `CALLBACK_URL_INVALID` (hostname not in allowlist).
- Callback success:
  - Command: `POST /api/v1/dedup/2d/search?...&callback_url=http://host.docker.internal:56874/hook`
  - Result: job `306a879b-d7f7-469f-a8b4-008da7066071` completed.
  - Job metadata: `callback_status=success`, `callback_attempts=1`, `callback_http_status=200`.
  - Callback payload received at `/tmp/dedup2d_callback.json`.
  - HMAC signature verified with `stage_secret`.

## Notes
- Callback server and dedupcad-vision were stopped after validation.
