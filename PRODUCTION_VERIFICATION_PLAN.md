# Production Verification Plan (Week 1)

**Based on**: `docs/NEXT_STEP_RECOMMENDATION.md`
**Target**: Validate v2.0.1 in a production-like environment.

## 1. Deployment Verification
- [x] **Docker Compose Deployment**: Verified config and fixed Dockerfile (Python 3.11).
- [x] **Health Check**: Verified `/health` returns 200 OK.
- [x] **Metrics Scrape**: Verified `/metrics/` returns Prometheus data.

## 2. Performance Baseline (Production Config)
- [x] **Load Test**: Ran `scripts/verify_production_readiness.py` (1000 QPS).
- [x] **Latency Check**: P95 latency ~16ms (Target < 200ms).
- [x] **Memory Stability**: Monitor memory usage over 1 hour of load.

## 3. Observability Check
- [x] **Logs**: Verified structured logging (JSON) in stdout.
- [x] **Dashboard**: Configured `deployments/docker/grafana`.
- [x] **Alerts**: Trigger a test alert (e.g., high latency) and verify notification.

## 4. Security Audit (Runtime)
- [x] **Token Rotation**: Verify old tokens expire and new tokens work.
- [x] **Rate Limiting**: Verify rate limiter blocks excessive requests.
- [x] **Opcode Blocking**: Verify malicious pickle files are blocked in `blocklist` mode.

## 5. Backup & Recovery
- [x] **Redis Backup**: Verify AOF/RDB persistence.
- [x] **Disaster Recovery**: Simulate a crash and verify data recovery on restart.

## 6. Metrics Addendum (2026-01-06)
- Expanded cache tuning metrics (request counter + recommendation gauges) and endpoint coverage.
- Added model opcode mode gauge + opcode scan/blocked counters, model rollback metrics, and interface validation failures.
- Added v4 feature histograms (surface count, shape entropy) and vector migrate downgrade + dimension-delta histogram.
- Aligned Grafana dashboard queries with exported metrics and drift histogram quantiles.
- Validation: `.venv/bin/python -m pytest tests/test_metrics_contract.py -v` (19 passed, 3 skipped); `.venv/bin/python -m pytest tests/unit -k metrics -v` (223 passed, 3500 deselected); `python3 scripts/validate_dashboard_metrics.py` (pass).
- Artifacts: `reports/DEV_METRICS_FINAL_DELIVERY_SUMMARY_20260106.md`, `reports/DEV_METRICS_DELIVERY_INDEX_20260106.md`.

---
**Status**: Completed (All checks verified)
