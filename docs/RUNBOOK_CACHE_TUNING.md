# Cache Tuning Runbook

Purpose: Safely adjust feature cache capacity/TTL and prewarm without destabilizing latency.

Endpoints:
- `POST /api/v1/health/cache/apply` {"capacity": <int>, "ttl_seconds": <int>} — apply new settings; opens 5‑minute rollback window.
- `POST /api/v1/health/cache/rollback` — revert to previous settings if window active.
- `POST /api/v1/health/cache/prewarm` — trigger prewarm logic (metrics, warm read path).

Metrics:
- `feature_cache_hits_total`, `feature_cache_miss_total` — hit ratio.
- `feature_cache_prewarm_total{status}` — prewarm invocations (success|skipped|error).
- `feature_cache_evictions_total` — churn indicator.
- `feature_cache_size` — current entry count.

Procedure:
1. Baseline: record hit ratio (5m, 30m), eviction rate, miss latency before change.
2. Apply: call apply endpoint; confirm response `status="applied"` and `can_rollback_until` populated.
3. Monitor: ensure hit ratio improves or remains stable; watch evictions (avoid sudden spike).
4. Prewarm (optional): call prewarm; verify metric increment and initial miss latency reduction.
5. Rollback (if needed): within window call rollback; confirm response `status="rolled_back"`.
6. Lock‑in: after window expiry confirm rollback returns `status="no_active_window"`.

Rollback Criteria:
- Hit ratio drops >15% vs baseline.
- Eviction rate > configured capacity * 0.5 per minute (indicates thrash).
- Miss latency p95 increases >2x.

Tuning Tips:
- Increase capacity gradually (e.g., +25%) rather than doubling.
- Introduce TTL only if stale data observed; start large (>1h) then tighten.
- Use prewarm after large capacity increase to reduce cold misses.

Alerts (optional future):
- Low hit ratio sustained (`rate(feature_cache_hits_total[15m]) / (hits+misses) < target`).
- High eviction rate spike.

References:
- `src/core/feature_cache.py`, `src/api/v1/health.py` (apply/rollback/prewarm logic).
- Metrics definitions: `src/utils/analysis_metrics.py`.
