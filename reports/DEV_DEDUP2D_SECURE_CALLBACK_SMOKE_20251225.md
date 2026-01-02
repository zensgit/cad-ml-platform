# DEV_DEDUP2D_SECURE_CALLBACK_SMOKE_20251225

## Scope

- Run a full dedup2d async smoke test with secure callback settings (HTTPS + allowlist).
- Validate health, metrics, job lifecycle, Redis keys, S3 cleanup, and HMAC webhook signature.

## Setup

- Callback tunnel: `https://data-globe-collaborative-receptor.trycloudflare.com/hook`
- Callback allowlist: `data-globe-collaborative-receptor.trycloudflare.com`
- Callback security:
  - `DEDUP2D_CALLBACK_ALLOW_HTTP=0`
  - `DEDUP2D_CALLBACK_BLOCK_PRIVATE_NETWORKS=1`
  - `DEDUP2D_CALLBACK_RESOLVE_DNS=1`
  - `DEDUP2D_CALLBACK_HMAC_SECRET=dedup2d-test`
- Vision target: `DEDUPCAD_VISION_URL=http://dedupcad-vision-api:8000`
- Buckets ensured in MinIO:
  - `dedup2d-uploads`
  - `dedupcad-drawings`
- DedupCAD Vision container:
  - Image: `dedupcad-vision:local`
  - Container: `dedupcad-vision-api` on `cad-ml-network`

## Test Steps

1. Health check: `GET /health`.
2. Metrics check: `GET /metrics/` (redirected from `/metrics`).
3. Negative callback test: submit async job with unallowlisted HTTPS callback
   (`https://example.com/hook`) and confirm 400.
4. Async job submission with HTTPS allowlisted callback.
5. Poll job until completion.
6. Verify job appears in `GET /api/v1/dedup/2d/jobs?limit=5`.
7. Verify Redis keys for job/payload/result and tenant job list.
8. Verify S3 cleanup (uploaded object removed).
9. Verify webhook delivery and HMAC signature match.

## Results

- Health: `status=healthy`.
- Metrics: `/metrics/` reachable with `dedup2d_*` metrics present.
- Negative callback test: HTTP 400 (allowlist enforced).
- Job:
  - Job id: `268d5d7b-cfa6-48b5-b19c-f40ef5ab19b8`
  - Status: `completed`
  - `total_matches=0`, `final_level=1`
- Callback:
  - `callback_status=success`, `callback_http_status=200`
  - HMAC signature verified (computed == header)
- Redis:
  - Keys present: `dedup2d:job:<job_id>`, `dedup2d:payload:<job_id>`, `dedup2d:result:<job_id>`
  - Tenant ZSET contains job id: `dedup2d:tenant:9f86d081884c7d65:jobs`
- S3 cleanup:
  - `mc stat` for `uploads/<job_id>/..._test_left.png` returned "Object does not exist".

## Validation Evidence

- Job response: `/tmp/dedup2d_job.json`
- Callback payload: `/tmp/dedup2d_callback_log.jsonl`
- HMAC check: computed signature matched header using `dedup2d-test`
- Redis dumps:
  - `/tmp/dedup2d_job_hash.txt`
  - `/tmp/dedup2d_payload.json`
  - `/tmp/dedup2d_result_value.json`
  - `/tmp/dedup2d_tenant_jobs.txt`
- Metrics snapshot: `/tmp/cad_ml_metrics.txt`
- Rejection response: `/tmp/dedup2d_reject.json`
- Temporary files were cleaned after the report was written.

## Notes

- The tunnel hostname is ephemeral (trycloudflare) and will change per run.
