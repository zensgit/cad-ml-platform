# DEV_DEDUP2D_ASYNC_SMOKE_20251225

## Scope

- Validate dedup2d async pipeline end-to-end with a live DedupCAD Vision service.
- Verify webhook callback delivery and HMAC signature validation.
- Confirm S3/MinIO storage works with the staging overrides.

## Setup

- DedupCAD Vision container built from `/Users/huazhou/Downloads/Github/dedupcad-vision`:
  - Image: `dedupcad-vision:local`
  - Container: `dedupcad-vision-api` on `cad-ml-network`
  - Env: `S3_ENDPOINT_URL=http://cad-ml-minio:9000`, `S3_BUCKET=dedupcad-drawings`,
    `REDIS_URL=redis://cad-ml-redis:6379/0`, `EVENT_BUS_ENABLED=false`
- Bucket created in MinIO: `dedupcad-drawings`
- cad-ml API/worker run with:
  - `DEDUPCAD_VISION_URL=http://dedupcad-vision-api:8000`
  - `DEDUP2D_CALLBACK_ALLOW_HTTP=1`
  - `DEDUP2D_CALLBACK_BLOCK_PRIVATE_NETWORKS=0`
  - `DEDUP2D_CALLBACK_HMAC_SECRET=dedup2d-test`

## Test Steps

1. Start local webhook receiver at `http://host.docker.internal:19080/hook`.
2. Submit async job:
   - Endpoint: `POST /api/v1/dedup/2d/search?mode=balanced&max_results=5&async=true&callback_url=...`
   - File: `data/dedupcad_batch_demo/test_left.png`
   - Header: `X-API-Key: test`
3. Poll job status until completion.
4. Validate webhook delivery and HMAC signature.

## Results

- Job status: `completed`
- Job id: `1264f151-7199-492a-896f-416d77dddaec`
- Result summary: `total_matches=0`, `final_level=1`
- Callback:
  - `callback_status=success`, `callback_http_status=200`
  - HMAC signature verified (computed == header)

## Validation Evidence

- Job response: `/tmp/dedup2d_job.json`
- Callback payload: `/tmp/dedup2d_callback_log.jsonl`
- HMAC check matched using `DEDUP2D_CALLBACK_HMAC_SECRET=dedup2d-test`

## Notes

- DedupCAD Vision reported indexes not ready (L1/L2 size 0), so no matches returned.
- DedupCAD Vision container left running for subsequent tests.
