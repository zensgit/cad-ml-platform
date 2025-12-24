# DEV_DEDUP2D_FINAL_SUMMARY_20251224

## Scope
- Final summary for Dedup2D staging + callback + network standardization work.

## Key Outcomes
- Dedup2D async pipeline validated end-to-end (API → Redis/ARQ → dedupcad-vision → results).
- S3/MinIO storage path validated with cleanup on finish.
- Callback allowlist + HMAC verified (rejects non-allowlisted host, signs payloads correctly).
- Compose overrides added for staging; runbook updated.
- Docker network unified as `cad-ml-network` to avoid cross-project DNS failures.
- Staging baseline restored (callback overrides removed), smoke checks passed.

## Reports Generated
- `reports/DEV_DEDUP2D_CHECKLIST_TESTS_20251224.md`
- `reports/DEV_DEDUP2D_E2E_WEBHOOK_20251224.md`
- `reports/DEV_DEDUP2D_E2E_MINIO_20251224.md`
- `reports/DEV_DEDUP2D_STAGING_RUNBOOK_20251224.md`
- `reports/DEV_DEDUP2D_STAGING_RUNBOOK_S3_20251224.md`
- `reports/DEV_DEDUP2D_CALLBACK_HMAC_ALLOWLIST_20251224.md`
- `reports/DEV_DEDUP2D_STAGING_COMPOSE_OVERRIDE_20251224.md`
- `reports/DEV_DEDUP2D_NETWORK_UNIFICATION_20251224.md`
- `reports/DEV_DEDUP2D_STAGING_SMOKE_AFTER_NETWORK_20251224.md`
- `reports/DEV_DEDUP2D_NETWORK_STANDARDIZED_20251224.md`
- `reports/DEV_DEDUP2D_CALLBACK_HMAC_AFTER_NETWORK_20251224.md`
- `reports/DEV_DEDUP2D_STAGING_BASELINE_AFTER_CALLBACK_20251224.md`

## Notable Decisions
- MinIO host ports left unexposed to avoid conflict with `dedupcad-minio` (internal-only access).
- S3 cleanup verification performed via `minio/mc` on `cad-ml-network`.

## Current State
- cad-ml services running on `cad-ml-network` with S3 backend and Redis async.
- Prometheus target for `cad-ml-api` is `up`.
- Grafana dashboard `Dedup2D Dashboard` visible.
