# Dedup2D Render Queue S3 E2E Validation (2025-12-19)

## Scope
- Repository: cad-ml-platform
- Goal: Validate render queue using S3/MinIO-backed file storage.
- Queue: `dedup2d:render` (production queue)

## Environment
- Container: `cad-ml-api`
- Redis: `redis://redis:6379/0`
- Vision: `http://host.docker.internal:58001`
- Storage: `DEDUP2D_FILE_STORAGE=s3`
- MinIO endpoint: `http://minio:9000`
- Bucket: `dedup2d-uploads`

## Input
- File: `/app/data/dedupcad_batch/input/BTJ01231501522-00短轴承座(盖)v2.dxf`
- Content type: `application/dxf`

## Steps
1. Submit job with `DEDUP2D_RENDER_QUEUE_NAME=dedup2d:render`.
2. Start render worker for queue `dedup2d:render` with S3 storage enabled.
3. Poll job status until completion.

## Result
- Job ID: `13979b4d-3191-4b96-ad61-834cedae8f6a`
- Status: `COMPLETED`
- Search result: `success=true`, `total_matches=8`, `final_level=2`
- Errors: none

## Notes
- File bytes stored in MinIO via S3 client and retrieved by render worker successfully.
- High‑fidelity rendering enabled (`DEDUPCAD2_RENDER_TEXT=1`, `DEDUPCAD2_RENDER_HATCH=1`).
