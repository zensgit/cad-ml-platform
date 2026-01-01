# DEV_DEDUP2D_STAGING_SMOKE_20260101

## Scope
- Validate dedup2d staging stack (API + worker + Redis + MinIO) against a running dedupcad-vision instance.
- Verify async job flow, worker execution, and observability signals.

## Environment
- dedupcad-vision: `http://localhost:8100`
- cad-ml API: `http://localhost:18000`
- Metrics: `http://localhost:18000/metrics/` (note `/metrics` redirects to `/metrics/`)
- Prometheus: `http://127.0.0.1:19091/-/healthy`
- MinIO: `http://localhost:19000` (console `http://localhost:19001`)

## Compose Startup
```bash
CAD_ML_API_PORT=18000 \
CAD_ML_API_METRICS_PORT=19090 \
CAD_ML_PROMETHEUS_PORT=19091 \
CAD_ML_ALERTMANAGER_PORT=19093 \
CAD_ML_GRAFANA_PORT=13001 \
CAD_ML_REDIS_PORT=16379 \
CAD_ML_MINIO_PORT=19000 \
CAD_ML_MINIO_CONSOLE_PORT=19001 \
DEDUPCAD_VISION_URL=http://host.docker.internal:8100 \
docker compose -f deployments/docker/docker-compose.yml \
  -f deployments/docker/docker-compose.minio.yml \
  -f deployments/docker/docker-compose.dedup2d-staging.yml up -d
```

MinIO init:
```bash
docker start -a cad-ml-minio-init
```

## Health Checks
API health:
```bash
curl -sSf --max-time 5 http://localhost:18000/health
```
Result: `status=healthy`, `redis=up`, `metrics_enabled=true`.

Dedupcad-vision proxy health:
```bash
curl -sSf --max-time 10 -H 'X-API-Key: test' \
  http://localhost:18000/api/v1/dedup/2d/health
```
Result: `status=healthy`, `service=caddedup-vision`, `version=0.2.0`.

Prometheus health (container confirmed):
```bash
curl -sSf --max-time 5 http://127.0.0.1:19091/-/healthy
```
Result: `Prometheus Server is Healthy.`

## Async Job Smoke Test
Request:
```bash
curl -sS -H 'X-API-Key: test' \
  -F 'file=@reports/eval_history/plots/combined_trend.png' \
  'http://localhost:18000/api/v1/dedup/2d/search?async=true'
```
Response:
```json
{"job_id":"5dbc9df9-8c52-4072-b5ef-396f5ddcdae2","status":"pending","poll_url":"/api/v1/dedup/2d/jobs/5dbc9df9-8c52-4072-b5ef-396f5ddcdae2","forced_async_reason":null}
```

Job status:
```bash
curl -sS -H 'X-API-Key: test' \
  http://localhost:18000/api/v1/dedup/2d/jobs/5dbc9df9-8c52-4072-b5ef-396f5ddcdae2 | jq '.'
```
Result (summary):
- `status=completed`
- `total_matches=0`, `duplicates=[]`, `similar=[]`
- `timing.total_ms=652.01`

Worker log excerpt:
```
05:10:11:   0.38s → 5dbc9df9-8c52-4072-b5ef-396f5ddcdae2:dedup2d_run_job('5dbc9df9-8c52-4072-b5ef-396f5ddcdae2')
05:10:12:   0.82s ← 5dbc9df9-8c52-4072-b5ef-396f5ddcdae2:dedup2d_run_job ● {'status': 'completed'}
```

## Metrics Verification
Dedup2d metrics:
```bash
curl -sSf --max-time 5 http://localhost:18000/metrics/ \
  | rg "dedup2d_" | head -n 20
```
Observed:
- `dedup2d_jobs_total{status="pending"}=1`
- `dedup2d_jobs_total{status="completed"}=1`
- `dedup2d_job_duration_seconds_count{status="completed"}=1`
- `dedup2d_job_queue_depth=0`

Dedupcad-vision client metrics:
```bash
curl -sSf --max-time 20 http://localhost:18000/metrics/ \
  | rg "dedupcad_vision_" | head -n 20
```
Observed:
- `dedupcad_vision_requests_total{endpoint="health",status="success"}=3`
- `dedupcad_vision_circuit_state{endpoint="health"}=0`

## Notes
- `GET /metrics` returns `307` redirect to `/metrics/` (expected with ASGI mount).
- Worker emits a `PythonDeprecationWarning` for boto3 on Python 3.9 (non-blocking).

## Result
- Dedup2d staging stack started successfully with S3/MinIO storage.
- Async job submission, worker execution, and metrics instrumentation verified.
