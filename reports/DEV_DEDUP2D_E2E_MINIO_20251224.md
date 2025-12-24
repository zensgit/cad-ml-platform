# DEV_DEDUP2D_E2E_MINIO_20251224

## Scope
- Run Dedup2D webhook E2E smoke test with MinIO S3 backend (Redis + ARQ + local fake vision + callback).

## Validation
- Command: `.venv/bin/python scripts/e2e_dedup2d_webhook_minio.py`
  - Result: `OK: job completed + callback received + signature verified + minio cleaned`.

## Notes
- Script starts a MinIO container and validates S3 cleanup after job completion.
