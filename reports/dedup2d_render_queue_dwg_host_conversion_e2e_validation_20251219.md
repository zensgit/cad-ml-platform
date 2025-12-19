# Dedup2D Render Queue DWG Conversion E2E Validation (2025-12-19)

## Scope
- Repository: cad-ml-platform
- Goal: Validate DWG conversion inside render worker (ODA) with high‑fidelity rendering.
- Queue: `dedup2d:render-e2e-host`

## Environment
- Worker: host `cad-ml-platform` venv (`.venv/bin/arq`)
- Redis: `redis://localhost:16379/0`
- Vision: `http://localhost:58001`
- Storage: `DEDUP2D_FILE_STORAGE=local`, `DEDUP2D_FILE_STORAGE_DIR=/Users/huazhou/Downloads/Github/cad-ml-platform/data/dedup2d_uploads`
- DWG Converter: `ODA_FILE_CONVERTER_EXE=/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter`

## Input
- DWG: `/Users/huazhou/Downloads/训练图纸/训练图纸/BTJ01231501522-00短轴承座(盖)v2.dwg`
- Content type: `image/vnd.dwg`

## Steps
1. Submit job to render queue `dedup2d:render-e2e-host`.
2. Start render worker on host with ODA converter available.
3. Poll job status until completion.

## Result
- Job ID: `228afb63-895a-460a-ab27-2c9cd90d3309`
- Status: `COMPLETED`
- Search result: `success=true`, `total_matches=8`, `final_level=2`
- Errors: none

## Worker Log Excerpt
```
17:02:07: Starting worker for 1 functions: dedup2d_run_job
17:02:07:  35.54s → 228afb63-895a-460a-ab27-2c9cd90d3309:dedup2d_run_job('228afb63-895a-460a-ab27-2c9cd90d3309') delayed=35.54s
17:02:17:   9.34s ← 228afb63-895a-460a-ab27-2c9cd90d3309:dedup2d_run_job ● {'status': 'completed'}
```
