# Dedup2D Production Go/No-Go Checklist

Use this to decide whether to promote staging -> production.

## Go criteria

- Vision reachable: DEDUPCAD_VISION_URL responds 200 /health
- Async job success: at least 3 staged jobs complete with expected Vision payload
- Job list works: GET /api/v1/dedup/2d/jobs returns completed jobs within TTL
- Worker healthy: no crash loops; queue depth stable (< max_jobs * 0.5)
- Metrics available: dedup2d_* metrics present in Prometheus
- Vision circuit closed: `dedupcad_vision_circuit_state` indicates CLOSED for health/search
- Alerts loaded: Dedup2D alert rules visible in Prometheus
- Dashboard visible: Grafana has "Dedup2D Dashboard"
- S3/MinIO access ok: upload + cleanup verified
- Callback policy ready: allowlist + HMAC secret configured if callbacks are enabled

## No-Go triggers

- Vision timeout rate > 5% in staging
- Vision circuit open or flapping (dedupcad_vision_circuit_state != 0)
- Job failure rate > 2% over 30 min
- Queue depth growing without recovery
- Callback failure rate > 5% (if callbacks enabled)
- Missing metrics or alerts after deployment
- S3 upload or cleanup failures
- Legacy payload fallback (`dedup2d_legacy_b64_fallback_total`) still increasing after rollout

## Rollback plan (fast)

1) Disable async backend (temporary)

```bash
DEDUP2D_ASYNC_BACKEND=inprocess
WORKERS=1
```

2) Scale worker to 0

- Set `dedup2d.worker.enabled=false` (Helm) or scale Deployment to 0

3) Revert release

- Roll back to previous Helm release revision

## Post-Go verification (first 1-2 hours)

- Confirm error rate and latency stay within baseline
- Confirm queue depth remains stable
- Confirm no spike in Vision timeouts
- Confirm job list still returns completed jobs
