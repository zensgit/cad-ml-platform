# METRICS_TEST_FIXTURE_REFACTOR_DESIGN

## Context
- Multiple tests duplicated metrics skip checks and helper functions.

## Decision
- Add shared fixtures in `tests/conftest.py`:
  - `require_metrics_enabled` to skip metric-only tests.
  - `metrics_text` callable to fetch `/metrics` text when enabled.
- Update affected tests to use the shared fixtures.

## Scope
- `tests/conftest.py`
- `tests/test_health_and_metrics.py`
- `tests/test_ocr_provider_down.py`
- `tests/test_vision_base64_rejection_reasons.py`
- `tests/test_ocr_invalid_mime.py`
- `tests/test_ocr_pdf_rejections.py`
- `tests/unit/test_analysis_cache_metrics.py`
- `tests/unit/test_drift_startup_trigger.py`
- `tests/unit/test_faiss_rebuild_backoff.py`
- `tests/unit/test_feature_cache_sliding_window.py`
- `tests/unit/test_parallel_execution_metric.py`
- `tests/unit/test_parallel_savings_metric.py`
- `tests/unit/test_model_security_validation.py`

## Rationale
- Centralizes the metrics-enabled gate to reduce drift across tests.
- Keeps non-metrics assertions active even when metrics are disabled.
