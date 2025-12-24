# DEV_DEDUP2D_STAGING_BASELINE_AFTER_CALLBACK_20251224

## Scope
- Revert callback test env to baseline staging config and run smoke checks.

## Actions
- Recreated `cad-ml-api` and `dedup2d-worker` without callback override.
- Ensured MinIO bucket `dedup2d-uploads` exists.

## Validation
- Health: `curl -sSf http://localhost:8000/health` → `status=healthy`.
- Submit async job:
  - Command: `curl -sSf -X POST "http://localhost:8000/api/v1/dedup/2d/search?mode=balanced&max_results=10&async=true" -H "X-API-Key: test" -F "file=@/tmp/dedup2d_sample.png;type=image/png"`
  - Result: `job_id=01733b4c-16de-4aef-8d92-4a0ea850024b`, `status=pending`.
- Poll job:
  - Result: `status=completed`.
- List jobs:
  - Command: `curl -sSf -H "X-API-Key: test" "http://localhost:8000/api/v1/dedup/2d/jobs?limit=5"`
  - Result: job present in list (total=4).
- Metrics:
  - `curl -sSL http://localhost:8000/metrics | rg dedup2d_` → dedup2d metrics present.
- S3 cleanup (MinIO):
  - `mc find myminio/dedup2d-uploads --name '*<job_id>*'` (via minio/mc on `cad-ml-network`) found no matches.
- Prometheus target:
  - `cad-ml-api` target health `up` via `http://localhost:9091/api/v1/targets`.
- Grafana:
  - `Dedup2D Dashboard` visible via `http://localhost:3001/api/search?query=Dedup2D`.

## Notes
- MinIO is internal only (no host ports); S3 verification used minio/mc on `cad-ml-network`.
- Local dedupcad-vision was stopped after validation.
