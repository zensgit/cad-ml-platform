# Dedup2D Go/No-Go Local Checks (2026-01-01)

## Scope

- Validate key local checks from `docs/DEDUP2D_PROD_GO_NO_GO.md` (vision health, job list, metrics, worker health).

## Commands

- `curl -s http://localhost:8100/health`
- `curl -s -L http://localhost:8000/api/v1/dedup/2d/jobs`
- `curl -s -L http://localhost:8000/metrics | rg -n "dedup2d_jobs_total|dedup2d_job_queue_depth|dedupcad_vision_requests_total"`
- `docker ps --format '{{.Names}}\t{{.Status}}' | rg 'cad-ml-api|cad-ml-dedup2d-worker'`

## Results

- Vision health: `200 OK` with service `caddedup-vision`.
- Job list: returns at least one completed job.
- Metrics: `dedup2d_*` counters/gauges and `dedupcad_vision_requests_total` present; queue depth `0.0`.
- Worker: `cad-ml-dedup2d-worker` running; API healthy.

## Notes

- Vision health reports indexes not ready; integration flags show `ml_platform.service_available=false`.
