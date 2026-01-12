# METRICS_DISABLED_UNIT_TEST_GUARDS_DESIGN

## Context
- Metrics-dependent unit tests fail when prometheus_client is unavailable and /metrics returns app_metrics_disabled.

## Decision
- Gate metric-specific unit assertions with metrics_enabled() from src.api.health_utils.
- Skip only metric-focused tests; leave functional behavior tests unchanged.

## Scope
- tests/unit/test_drift_startup_trigger.py
- tests/unit/test_faiss_rebuild_backoff.py
- tests/unit/test_feature_cache_sliding_window.py
- tests/unit/test_model_security_validation.py (metric-only checks)
- tests/unit/test_parallel_execution_metric.py
- tests/unit/test_parallel_savings_metric.py

## Rationale
- Avoid false negatives in reduced environments while preserving coverage where metrics are available.
