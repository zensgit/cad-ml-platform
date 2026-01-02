# DEV_DEDUP2D_SECURE_CALLBACK_SCRIPT_20251225

## Scope

- Add a secure callback smoke script that enforces HTTPS + allowlist + HMAC.
- Update the staging runbook with secure callback instructions.
- Execute a secure callback smoke run and capture evidence.

## Changes

- Added `scripts/e2e_dedup2d_secure_callback.sh` to automate the secure callback smoke flow
  (cloudflared tunnel, allowlist enforcement, HMAC validation, Redis/S3 checks).
- Updated `docs/DEDUP2D_STAGING_RUNBOOK.md` with the secure HTTPS callback section.

## Validation

- Command: `DEDUPCAD_VISION_START=1 scripts/e2e_dedup2d_secure_callback.sh`
- Results:
  - Job id: `c665aaea-df4e-42cf-b8e2-301e1e586ca6` (status: `completed`)
  - Callback: `callback_status=success`, `callback_http_status=200`
  - Allowlist rejection: HTTPS callback to `example.com` returned HTTP 400
  - HMAC signature verified (secret: `dedup2d-test`)
  - Metrics: `dedup2d_jobs_total` present on `/metrics/`
  - Redis keys present: `dedup2d:job:<job_id>`, `dedup2d:payload:<job_id>`,
    `dedup2d:result:<job_id>`, `dedup2d:tenant:9f86d081884c7d65:jobs`
  - S3 cleanup: uploaded object not found (`mc stat` reported "Object does not exist")

## Artifacts

- `/tmp/dedup2d_secure_callback.SPNSq2` (cloudflared log, callback payload, metrics snapshot)

## Notes

- The trycloudflare hostname is ephemeral per run.
