# Dedup2D Staging Runbook

Scope: bring up cad-ml-platform + dedup2d-worker + Redis + S3/MinIO and validate end-to-end
against a real dedupcad-vision instance.

## Prerequisites

- dedupcad-vision reachable at DEDUPCAD_VISION_URL
- Redis reachable
- S3/MinIO reachable (bucket + credentials)
- staging API key for cad-ml-platform (X-API-Key)

## 1) Deploy order (recommended)

1. Deploy Redis + S3/MinIO
2. Deploy dedupcad-vision and confirm /health
3. Deploy cad-ml-platform API (DEDUP2D_ASYNC_BACKEND=redis)
4. Deploy dedup2d-worker (ARQ)
5. Enable Prometheus scrape + Grafana dashboard

## 1.5) Optional local staging via Docker Compose

If you run staging locally, use the compose overrides:

```bash
docker compose -f deployments/docker/docker-compose.yml \
  -f deployments/docker/docker-compose.minio.yml \
  -f deployments/docker/docker-compose.dedup2d-staging.yml up -d
```

Recommended env (or .env) for the override:

- DEDUP2D_S3_BUCKET, DEDUP2D_S3_ENDPOINT, DEDUP2D_S3_REGION
- DEDUP2D_S3_ACCESS_KEY, DEDUP2D_S3_SECRET_KEY (or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY)
- DEDUP2D_CALLBACK_ALLOWLIST, DEDUP2D_CALLBACK_HMAC_SECRET
- DEDUP2D_CALLBACK_ALLOW_HTTP (set to 1 only for local dev)
- DEDUP2D_CALLBACK_BLOCK_PRIVATE_NETWORKS, DEDUP2D_CALLBACK_RESOLVE_DNS
- CAD_ML_MINIO_PORT, CAD_ML_MINIO_CONSOLE_PORT (optional host ports, e.g. 19000/19001 if
  9000/9001 are already in use)

Note: `deployments/docker/docker-compose.yml` pins the network name to `cad-ml-network`
so different compose project names can share a single network. If you already have
containers on the old prefixed network, recreate them or attach the `cad-ml-network`
manually.

## 2) Config checklist (staging)

- DEDUP2D_ASYNC_BACKEND=redis
- DEDUP2D_REDIS_URL=redis://<redis-host>:6379/0
- DEDUP2D_REDIS_KEY_PREFIX=dedup2d
- DEDUP2D_ARQ_QUEUE_NAME=dedup2d:queue
- DEDUP2D_FILE_STORAGE=s3
- DEDUP2D_S3_ENDPOINT, DEDUP2D_S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
- DEDUPCAD_VISION_URL=http://<vision-host>:58001
- DEDUPCAD_VISION_TIMEOUT_SECONDS=60
- DEDUP2D_ASYNC_MAX_JOBS, DEDUP2D_ASYNC_TTL_SECONDS, DEDUP2D_ASYNC_JOB_TIMEOUT_SECONDS

## 3) Smoke tests (API)

### Health

- GET /health (expect status=healthy)
- GET /metrics (expect dedup2d_* metrics present)

### Submit async job

```bash
curl -sSf -X POST \
  "http://<api-host>/api/v1/dedup/2d/search?mode=balanced&max_results=10&async=true" \
  -H "X-API-Key: <api-key>" \
  -F "file=@<png-or-jpg-file>;type=image/png"
```

Expect:

```json
{"job_id":"...","status":"pending","poll_url":"/api/v1/dedup/2d/jobs/<job_id>"}
```

### Poll job

```bash
curl -sSf -H "X-API-Key: <api-key>" \
  "http://<api-host>/api/v1/dedup/2d/jobs/<job_id>"
```

Expect `status=completed` and `result` with Vision payload.

### List jobs

```bash
curl -sSf -H "X-API-Key: <api-key>" \
  "http://<api-host>/api/v1/dedup/2d/jobs?limit=5"
```

Expect completed job to appear within TTL.

### Callback allowlist + HMAC (optional)

Allowlist rejection (should return 400):

```bash
curl -sSf -X POST \
  "http://<api-host>/api/v1/dedup/2d/search?mode=balanced&max_results=10&async=true&callback_url=http://example.com/hook" \
  -H "X-API-Key: <api-key>" \
  -F "file=@<png-or-jpg-file>;type=image/png"
```

Callback success (use an allowlisted host):

```bash
curl -sSf -X POST \
  "http://<api-host>/api/v1/dedup/2d/search?mode=balanced&max_results=10&async=true&callback_url=https://<callback-host>/hook" \
  -H "X-API-Key: <api-key>" \
  -F "file=@<png-or-jpg-file>;type=image/png"
```

Then poll the job and confirm:
- callback_status=success
- callback_http_status=200
- HMAC signature header (X-Dedup-Signature) validates with DEDUP2D_CALLBACK_HMAC_SECRET

## 4) Worker validation

- Confirm ARQ worker logs show job execution and no uncaught exceptions
- Confirm Redis keys exist:
  - dedup2d:job:<job_id>
  - dedup2d:result:<job_id>
  - dedup2d:tenant:<tenant_id>:jobs

## 5) Monitoring validation

- Prometheus target for cad-ml-platform is UP
- Grafana dashboard "Dedup2D Dashboard" exists and shows non-zero samples
- No new alert rules firing unexpectedly

## 6) Failure handling

If job fails:

- Check worker logs for Vision HTTP errors or timeouts
- Confirm DEDUPCAD_VISION_URL is reachable from worker Pod
- Confirm S3 bucket permissions; verify cleanup_on_finish if enabled

## 7) Exit criteria

- At least one async job completes successfully with a real Vision response
- Job appears in GET /api/v1/dedup/2d/jobs
- Metrics are scraped; dashboard shows data
