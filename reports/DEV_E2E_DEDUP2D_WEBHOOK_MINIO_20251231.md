# Dedup2D Webhook E2E Smoke (MinIO) (2025-12-31)

## Scope

- Run Dedup2D webhook E2E smoke with Redis + ARQ + MinIO-backed storage.

## Command

- `./.venv/bin/python scripts/e2e_dedup2d_webhook_minio.py`

## Results

- OK: job completed, callback received, signature verified, MinIO cleaned.

## Notes

- Script uses ephemeral ports, temporary directories, and removes the MinIO bucket on exit.
