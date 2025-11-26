# Deployment Staging Checklist

## Prechecks
- Configure headers: `X-API-Key` (service key) and `X-Admin-Token` (admin operations).
- Verify environment variables:
  - Recovery: `FAISS_RECOVERY_FLAP_THRESHOLD`, `FAISS_RECOVERY_FLAP_WINDOW_SECONDS`, `FAISS_RECOVERY_SUPPRESSION_SECONDS`, `FAISS_RECOVERY_STATE_PATH`.
  - Cache: capacity/TTL defaults; ensure 5-minute rollback window expectations.
  - Security: `ALLOWED_MODEL_HASHES`, `MODEL_MAX_MB`, `MODEL_OPCODE_MODE`.
- Prometheus:
  - Scrape target healthy, rules loaded: `promtool check rules prometheus/rules/cad_ml_phase5_alerts.yaml`.
- Grafana:
  - Datasource OK; dashboard `grafana/dashboards/observability.json` imported.

## Deploy
- Roll out to staging environment.
- Health probe: `GET /health` returns 200; `/metrics` endpoint reachable.
- Warm-up: run a few analyze calls to populate v4 metrics.

## Canary (5–10%)
- Monitor SLOs:
  - v4 p95 vs v3 p95 ratio ≤ 2× (target) / ≤ 4× (temporary).
  - Cache hit ratio ≥ 70% temporary baseline.
  - Recovery duration ≤ 5 minutes temporary, aiming ≤ 2 minutes.

## Validation
- Health: `GET /api/v1/health/faiss/health` returns keys `degraded`, `degradation_history_count`, `next_recovery_eta`, `manual_recovery_in_progress`.
- Metrics: confirm updates for
  - `feature_extraction_latency_seconds{version="v4"}`
  - `similarity_degraded_total{event}`
  - `faiss_recovery_attempts_total{result}`
  - `faiss_degraded_duration_seconds`
- Alerts:
  - Trigger synthetic degradation to see `VectorStoreDegraded`.
  - Lower cache hit temporarily to test `LowCacheHitRatio`.
  - Exercise blocked opcode to ensure `OpcodeWhitelistViolation`.
- Dashboard:
  - Panels render: degraded/restored, recovery attempts, ETA gauge, v4 p95, cache hit/miss, opcode audit/violations, migration delta.

## Smoke Tests
- Migration preview: `GET /api/v1/vectors/migrate/preview?to_version=v4&limit=5` returns `avg_delta`, `median_delta`, `warnings`.
- Cache controls:
  - Apply → response includes snapshot/previous.
  - Prewarm → success response.
  - Rollback → only within 5-minute window.
- Model reload:
  - Requires `X-API-Key` + `X-Admin-Token`.
  - Failure responses use structured error with `code`, `stage`, `message`, `context`.

## Rollback
- Application: revert deployment via orchestrator.
- Cache: use `POST /api/v1/health/features/cache/rollback` within window.
- Recovery: if flapping, increase suppression via env and use `POST /api/v1/faiss/recover`.

## References
- Metrics: `src/utils/analysis_metrics.py`
- Health & recovery: `src/api/v1/health.py`, `src/core/similarity.py`
- Vectors & preview: `src/api/v1/vectors.py`
- Model reload & security: `src/ml/classifier.py`, `src/api/v1/model.py`
- Alerts/Dashboard: `prometheus/rules/cad_ml_phase5_alerts.yaml`, `grafana/dashboards/observability.json`
- Runbook: `docs/RUNBOOK_FLAPPING.md`
