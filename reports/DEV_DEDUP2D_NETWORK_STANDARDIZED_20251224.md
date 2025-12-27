# DEV_DEDUP2D_NETWORK_STANDARDIZED_20251224

## Scope
- Standardize cad-ml services on `cad-ml-network` and re-run staging smoke.

## Actions
- Removed running containers: `cad-ml-api`, `cad-ml-dedup2d-worker`, `cad-ml-redis`, `cad-ml-minio`, `cad-ml-prometheus`, `cad-ml-grafana`.
- Recreated stack using compose + staging overrides; kept `minio` internal (no host ports) due to port 9000 conflict with `dedupcad-minio`.
- Ensured MinIO bucket `dedup2d-uploads` exists (created via boto3).

## Validation
- Health: `curl -sSf http://localhost:8000/health` → `status=healthy`.
- Submit async job:
  - Command: `curl -sSf -X POST "http://localhost:8000/api/v1/dedup/2d/search?mode=balanced&max_results=10&async=true" -H "X-API-Key: test" -F "file=@/tmp/dedup2d_sample.png;type=image/png"`
  - Result: `job_id=1ddcb608-6d13-4d9f-ac16-fd53173a5c9e`, `status=pending`.
- Poll job:
  - Result: `status=completed`.
- List jobs:
  - Command: `curl -sSf -H "X-API-Key: test" "http://localhost:8000/api/v1/dedup/2d/jobs?limit=5"`
  - Result: `total=2`, new job present in list.
- Metrics:
  - `curl -sSL http://localhost:8000/metrics | rg dedup2d_` → dedup2d metrics present.
- S3 cleanup (MinIO):
  - `boto3 list_objects_v2` under `dedup2d-uploads/uploads` found no keys for job id.
- Prometheus target:
  - `cad-ml-api` target health `up` via `http://localhost:9091/api/v1/targets`.
- Grafana:
  - `Dedup2D Dashboard` visible via `http://localhost:3001/api/search?query=Dedup2D`.

## Notes
- MinIO host ports 9000/9001 were not exposed due to conflicts with `dedupcad-minio`; access via container network only.
- Local dedupcad-vision was stopped after validation.
