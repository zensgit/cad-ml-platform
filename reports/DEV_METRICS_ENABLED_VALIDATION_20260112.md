# DEV_METRICS_ENABLED_VALIDATION_20260112

## Setup
- python3.13 -m pip install --user --break-system-packages prometheus_client

## Commands
- pytest tests/test_metrics_contract.py -v
- pytest tests/unit/test_drift_startup_trigger.py tests/unit/test_faiss_rebuild_backoff.py tests/unit/test_feature_cache_sliding_window.py tests/unit/test_parallel_execution_metric.py tests/unit/test_parallel_savings_metric.py tests/unit/test_analysis_cache_metrics.py tests/unit/test_similarity_degraded_metrics.py -v
- pytest tests/unit/test_model_security_validation.py -k "model_security_fail_metric" -v

## Results
- metrics_contract: 19 passed, 3 skipped in 5.72s (fallback-when-disabled and strict-mode checks skipped)
- metrics_unit_targets: 9 passed in 2.24s
- model_security_metrics: 2 passed, 14 deselected in 0.68s
