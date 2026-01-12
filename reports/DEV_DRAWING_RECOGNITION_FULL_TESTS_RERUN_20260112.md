# DEV_DRAWING_RECOGNITION_FULL_TESTS_RERUN_20260112

## Scope
Re-run full test suite after installing PyJWT for the Python 3.13 environment.

## Setup
- Installed PyJWT for python3.13:

```bash
/opt/homebrew/opt/python@3.13/bin/python3.13 -m pip install --user --break-system-packages PyJWT
```

## Command
```bash
pytest tests -v --cov=src --cov-report=term-missing
```

## Result
- **23 failed**, **3966 passed**, **91 skipped**.

## Failures
- tests/test_health_and_metrics.py::test_metrics_has_vision_and_ocr_counters
- tests/test_health_and_metrics.py::test_metrics_rejected_counter_for_large_base64
- tests/test_ocr_provider_down.py::test_ocr_provider_down
- tests/test_vision_base64_rejection_reasons.py::test_vision_base64_invalid_char_reason
- tests/test_vision_base64_rejection_reasons.py::test_vision_base64_padding_error_reason
- tests/test_vision_base64_rejection_reasons.py::test_vision_base64_too_large_reason
- tests/unit/test_analysis_cache_metrics.py::test_analysis_cache_hit_miss_metrics
- tests/unit/test_confidence_calibrator_coverage.py::TestPlattScaling::test_platt_scaling_initialization
- tests/unit/test_confidence_calibrator_coverage.py::TestPlattScaling::test_platt_scaling_calibrate_unfitted_returns_raw
- tests/unit/test_confidence_calibrator_coverage.py::TestIsotonicCalibration::test_isotonic_initialization
- tests/unit/test_confidence_calibrator_coverage.py::TestIsotonicCalibration::test_isotonic_calibrate_unfitted_returns_raw
- tests/unit/test_drift_startup_trigger.py::test_drift_startup_trigger_metric_present
- tests/unit/test_faiss_rebuild_backoff.py::test_faiss_rebuild_backoff_metric
- tests/unit/test_feature_cache_sliding_window.py::test_feature_cache_sliding_window_metrics
- tests/unit/test_model_security_validation.py::test_model_security_fail_metric_magic_invalid
- tests/unit/test_model_security_validation.py::test_model_security_fail_metric_hash_mismatch
- tests/unit/test_ocr_endpoint_coverage.py::TestOcrExtractEndpoint::test_ocr_extract_success
- tests/unit/test_ocr_endpoint_coverage.py::TestOcrExtractEndpoint::test_ocr_extract_success_with_idempotency
- tests/unit/test_ocr_endpoint_coverage.py::TestOcrExtractEndpoint::test_ocr_extract_outer_http_exception
- tests/unit/test_parallel_execution_metric.py::test_parallel_execution_gauge
- tests/unit/test_parallel_savings_metric.py::test_parallel_savings_metric_observed
- tests/unit/test_similarity_degraded_metrics.py::test_similarity_degraded_metric_increment_on_degrade
- tests/unit/test_similarity_degraded_metrics.py::test_similarity_restored_metric_increment_on_recovery
