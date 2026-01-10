# DEV_DEDUP2D_STAGING_RUNBOOK_20251224

## Scope
- Execute Dedup2D staging runbook against a real dedupcad-vision instance.

## Setup
- Started dedupcad-vision locally on `http://127.0.0.1:58001` with `S3_ENABLED=false`, `EVENT_BUS_ENABLED=false`.
- Connected Docker networks to restore Redis + Prometheus DNS:
  - `docker network connect --alias redis docker_cad-ml-network cad-ml-redis`
  - `docker network connect --alias cad-ml-api cad-ml-platform-phase4_cad-ml-network cad-ml-api`
- `DEDUP2D_FILE_STORAGE=local` in `cad-ml-api` (MinIO reachable but not used in this run).

## Validation
- Health: `curl -sSf http://localhost:8000/health` → `status=healthy`.
- Metrics: `curl -sSL http://localhost:8000/metrics | rg dedup2d_` → dedup2d metrics present.
- Submit async job:
  - Command: `curl -sSf -X POST "http://localhost:8000/api/v1/dedup/2d/search?mode=balanced&max_results=10&async=true" -H "X-API-Key: test" -F "file=@/tmp/dedup2d_sample.png;type=image/png"`
  - Result: `job_id=96c3f0a6-0f8c-417b-9c33-e8e9a50db2d7`, `status=pending`.
- Poll job:
  - Result: `status=completed`, result keys present (duplicates/similar/timing).
- List jobs:
  - Command: `curl -sSf -H "X-API-Key: test" "http://localhost:8000/api/v1/dedup/2d/jobs?limit=5"`
  - Result: `total=1`, job present in `jobs` list.
- Worker validation:
  - Logs show job completion: `dedup2d_run_job('96c3f0a6-0f8c-417b-9c33-e8e9a50db2d7')` → `completed`.
- Redis keys:
  - `dedup2d:job:96c3f0a6-0f8c-417b-9c33-e8e9a50db2d7` exists.
  - `dedup2d:result:96c3f0a6-0f8c-417b-9c33-e8e9a50db2d7` exists.
  - `dedup2d:tenant:9f86d081884c7d65:jobs` contains job id.
- Prometheus target:
  - `cad-ml-api` target health `up` via `http://localhost:19091/api/v1/targets`.
- Grafana:
  - `Dedup2D Dashboard` visible via `http://localhost:3001/api/search?query=Dedup2D` (auth `admin:admin`).

## Notes
- MinIO health: `curl -sSf http://localhost:19000/minio/health/ready` succeeded.
- dedupcad-vision was stopped after validation.
