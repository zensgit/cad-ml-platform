# DEV_METRICS_TEST_FIXTURE_REFACTOR_20260112

## Summary
- Centralized metrics skip logic via shared fixtures and updated affected tests.

## Tests
- pytest tests/test_health_and_metrics.py tests/test_ocr_provider_down.py tests/test_vision_base64_rejection_reasons.py tests/test_ocr_invalid_mime.py tests/test_ocr_pdf_rejections.py tests/unit/test_drift_startup_trigger.py tests/unit/test_faiss_rebuild_backoff.py tests/unit/test_feature_cache_sliding_window.py tests/unit/test_parallel_execution_metric.py tests/unit/test_parallel_savings_metric.py tests/unit/test_analysis_cache_metrics.py tests/unit/test_model_security_validation.py -v

## Results
- 32 passed in 6.94s
