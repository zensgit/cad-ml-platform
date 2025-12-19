# Dedup2D Render Queue DWG E2E Validation (2025-12-19)

## Scope
- Repository: cad-ml-platform
- Goal: Validate DWG (converted to DXF) flow through render queue with high‑fidelity rendering.
- Queue: `dedup2d:render-e2e`

## Environment
- Container: `cad-ml-api`
- Redis: `redis://redis:6379/0`
- Vision: `http://host.docker.internal:58001`
- Storage: `DEDUP2D_FILE_STORAGE=local`, `DEDUP2D_FILE_STORAGE_DIR=/app/data/dedup2d_uploads`

## Input
- Original DWG: `/Users/huazhou/Downloads/训练图纸/训练图纸/BTJ01231501522-00短轴承座(盖)v2.dwg`
- DXF used in container: `/app/data/dedupcad_batch/input/BTJ01231501522-00短轴承座(盖)v2.dxf`
- Content type: `application/dxf`

## Steps
1. Submit job with `DEDUP2D_RENDER_QUEUE_NAME=dedup2d:render-e2e`.
2. Start render worker for queue `dedup2d:render-e2e` with high‑fidelity settings:
   - `DEDUPCAD2_RENDER_TEXT=1`
   - `DEDUPCAD2_RENDER_HATCH=1`
   - `DEDUPCAD2_RENDER_FALLBACK=1`
3. Poll job status.

## Result
- Job ID: `a26db37a-1371-4b75-81d9-a4e7fdc15a70`
- Status: `COMPLETED`
- Search result: `success=true`, `total_matches=8`, `final_level=2`
- Errors: none

## Notes
- DXF artifact already existed from prior ODA conversion; no additional conversion was needed in this run.
- Render worker processed the job in ~1.96s.
