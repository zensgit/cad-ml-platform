# Dedup2D PR Description (cad-ml-platform)

Title:

- feat(dedup2d): productionize async 2d dedup pipeline

Summary:

- Add Redis + ARQ async job backend with rolling upgrade compatibility
- Add file storage abstraction (local/S3) and GC tooling
- Add webhook callback flow with SSRF guardrails + HMAC support
- Add observability assets (metrics, alerts, Grafana dashboard)
- Add Helm templates for worker + GC CronJob
- Fix docker-compose runtime issues (port conflicts, network subnet overlap)
- Fix Redis job listing so completed jobs are visible within TTL

Testing:

- ./.venv/bin/python -m pytest -q
- ./.venv/bin/python -m pytest -q tests/unit/test_dedup2d_job_list_redis.py
- ./.venv/bin/python scripts/e2e_dedup2d_webhook.py
- ./.venv/bin/python scripts/e2e_dedup2d_webhook_minio.py

Config changes:

- DEDUP2D_ASYNC_BACKEND=redis
- DEDUP2D_REDIS_URL, DEDUP2D_REDIS_KEY_PREFIX, DEDUP2D_ARQ_QUEUE_NAME
- DEDUP2D_FILE_STORAGE=local|s3 and related S3 vars
- DEDUP2D_ASYNC_MAX_JOBS, DEDUP2D_ASYNC_TTL_SECONDS, DEDUP2D_ASYNC_JOB_TIMEOUT_SECONDS
- DEDUPCAD_VISION_URL, DEDUPCAD_VISION_TIMEOUT_SECONDS
- Webhook vars: DEDUP2D_CALLBACK_* (allowlist, HMAC, timeouts)

Rollout notes:

- Recommended order: deploy new workers -> optionally enable bytes_b64 dual-write -> deploy API -> disable bytes_b64
- In prod, use S3 storage (no shared disk) and dedicated worker Deployment

Docs:

- reports/dedup2d_phase4_verification_report_20251219.md
- reports/dedup2d_cross_repo_integration_report_20251219.md
- docs/DEDUP2D_PR_AND_PROD_CHECKLIST.md
- docs/DEDUP2D_STAGING_RUNBOOK.md
- docs/DEDUP2D_PROD_GO_NO_GO.md
- docs/DEDUP2D_VISION_INTEGRATION_CONTRACT.md
