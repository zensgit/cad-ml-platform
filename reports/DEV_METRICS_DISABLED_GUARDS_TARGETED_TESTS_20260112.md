# DEV_METRICS_DISABLED_GUARDS_TARGETED_TESTS_20260112

## Summary
- Metrics-dependent unit tests skip cleanly when metrics are disabled.

## Commands
- pytest tests/unit/test_drift_startup_trigger.py tests/unit/test_faiss_rebuild_backoff.py tests/unit/test_feature_cache_sliding_window.py tests/unit/test_parallel_execution_metric.py tests/unit/test_parallel_savings_metric.py -v
- pytest tests/unit/test_model_security_validation.py -k "model_security_fail_metric" -v

## Results
- 5 skipped in 1.56s
- 2 skipped, 14 deselected in 0.62s
