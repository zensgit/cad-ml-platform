# DEV_DEDUP2D_STAGING_RUNBOOK_S3_20251224

## Scope
- Execute Dedup2D staging runbook using S3/MinIO backend.

## Setup
- Recreated `cad-ml-api` + `dedup2d-worker` with S3 backend:
  - `DEDUP2D_FILE_STORAGE=s3`, `DEDUP2D_S3_ENDPOINT=http://minio:9000`, bucket `dedup2d-uploads`, prefix `uploads`.
- Connected MinIO to `docker_cad-ml-network` (alias `minio`) and re-attached `cad-ml-api` to Prometheus network.
- Ensured MinIO bucket exists via `minio/mc`.
- Started local dedupcad-vision on `http://127.0.0.1:58001` (S3/event bus disabled).

## Validation
- Health: `curl -sSf http://localhost:8000/health` → `status=healthy`.
- Metrics: `curl -sSL http://localhost:8000/metrics | rg dedup2d_` → dedup2d metrics present.
- Submit async job:
  - Command: `curl -sSf -X POST "http://localhost:8000/api/v1/dedup/2d/search?mode=balanced&max_results=10&async=true" -H "X-API-Key: test" -F "file=@/tmp/dedup2d_sample.png;type=image/png"`
  - Result: `job_id=458dcd41-df69-47c4-bb0e-b017ddacf191`, `status=pending`.
- Poll job:
  - Result: `status=completed`, result keys present (duplicates/similar/timing).
- List jobs:
  - Command: `curl -sSf -H "X-API-Key: test" "http://localhost:8000/api/v1/dedup/2d/jobs?limit=5"`
  - Result: `total=2`, job present in `jobs` list.
- Worker logs:
  - `dedup2d_run_job('458dcd41-df69-47c4-bb0e-b017ddacf191')` completed.
- Redis keys:
  - `dedup2d:job:458dcd41-df69-47c4-bb0e-b017ddacf191` exists.
  - `dedup2d:result:458dcd41-df69-47c4-bb0e-b017ddacf191` exists.
  - `dedup2d:tenant:9f86d081884c7d65:jobs` contains job id.
- S3 cleanup check (MinIO):
  - `boto3 list_objects_v2` under `dedup2d-uploads/uploads` found no keys for job id.
- Prometheus target:
  - `cad-ml-api` target health `up` via `http://localhost:19091/api/v1/targets`.
- Grafana:
  - `Dedup2D Dashboard` visible via `http://localhost:3001/api/search?query=Dedup2D` (auth `admin:admin`).

## Notes
- Worker logs warn: boto3 will deprecate Python 3.9 in 2026 (informational).
- dedupcad-vision was stopped after validation.
