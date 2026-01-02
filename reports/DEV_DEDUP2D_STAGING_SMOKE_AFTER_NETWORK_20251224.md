# DEV_DEDUP2D_STAGING_SMOKE_AFTER_NETWORK_20251224

## Scope
- Recreate cad-ml API/worker on `cad-ml-network` and re-run staging smoke checks.

## Setup
- Attempted full compose recreate; base containers already existed (name conflicts). Recreated only:
  - `cad-ml-api`
  - `cad-ml-dedup2d-worker`
- Attached existing Redis and MinIO containers to `cad-ml-network`.
- Prometheus target initially down due to network split; re-attached `cad-ml-api` to the Prometheus network.
- Started local dedupcad-vision on `http://127.0.0.1:58001`.

## Validation
- Health: `curl -sSf http://localhost:8000/health` → `status=healthy`.
- Submit async job:
  - Command: `curl -sSf -X POST "http://localhost:8000/api/v1/dedup/2d/search?mode=balanced&max_results=10&async=true" -H "X-API-Key: test" -F "file=@/tmp/dedup2d_sample.png;type=image/png"`
  - Result: `job_id=ea6fe3a1-6d41-4982-b500-9cfeed5bbc1b`, `status=pending`.
- Poll job:
  - Result: `status=completed`.
- List jobs:
  - Command: `curl -sSf -H "X-API-Key: test" "http://localhost:8000/api/v1/dedup/2d/jobs?limit=5"`
  - Result: `total=4`, new job present in list.
- Metrics:
  - `curl -sSL http://localhost:8000/metrics | rg dedup2d_` → dedup2d metrics present.
- S3 cleanup (MinIO):
  - `boto3 list_objects_v2` under `dedup2d-uploads/uploads` found no keys for job id.
- Prometheus target:
  - `cad-ml-api` target health `up` after re-attaching to Prometheus network.
- Grafana:
  - `Dedup2D Dashboard` visible via API search.

## Notes
- Prometheus container still resides on legacy prefixed network; full network unification requires recreating base services.
- dedupcad-vision was stopped after validation.
