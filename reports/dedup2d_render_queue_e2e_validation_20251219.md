# Dedup2D Render Queue E2E Validation (2025-12-19)

## Scope
- Repository: cad-ml-platform
- Goal: Verify CAD jobs are routed to the render queue and completed by the render worker.
- Queue: `dedup2d:render-e2e` (isolated from existing jobs)

## Environment
- Container: `cad-ml-api`
- Redis: `redis://redis:6379/0`
- Vision: `http://host.docker.internal:58001`
- Storage: `DEDUP2D_FILE_STORAGE=local`, `DEDUP2D_FILE_STORAGE_DIR=/app/data/dedup2d_uploads`

## Test Input
- File: `/app/data/dedupcad_batch/input/BTJ01230901522-00汽水分离器v2.dxf`
- Content type: `application/dxf`
- Request params: `mode=balanced`, `max_results=5`, `enable_precision=false`

## Steps
1. Submit job with `DEDUP2D_RENDER_QUEUE_NAME=dedup2d:render-e2e`.
2. Start render worker with `DEDUP2D_ARQ_QUEUE_NAME=dedup2d:render-e2e` and high‑fidelity render settings.
3. Poll job status.

## Result
- Job ID: `ae37f3b2-c84f-4c5b-8058-576767a44c8a`
- Worker output:
  - `dedup2d_run_job` completed in ~1.47s
- Job status: `COMPLETED`
- Search result: `success=true`, `total_matches=0`, `final_level=1`
- Errors: none

## Notes
- Render worker ran with `DEDUPCAD2_RENDER_TEXT=1` and `DEDUPCAD2_RENDER_HATCH=1`.
- Render queue isolation avoids legacy jobs with mismatched storage backend.
