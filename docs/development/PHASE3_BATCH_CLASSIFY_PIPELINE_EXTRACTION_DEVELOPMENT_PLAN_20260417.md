# Phase 3 Batch Classify Pipeline Extraction Development Plan

## Goal

Extract the `/api/v1/analyze/batch-classify` route orchestration into a shared
helper while preserving:

- V16 batch classification behavior
- V6 sequential fallback behavior
- fine/coarse batch classification contract fields
- request/response schema and metrics behavior

## Scope

### In

- add `src/core/classification/batch_classify_pipeline.py`
- move temp-file preparation, classifier dispatch, result aggregation, and
  metrics recording into the shared helper
- keep `batch_classify` as a thin route delegate
- add helper-level and route-level regression coverage
- fix valid-result index alignment when unsupported files appear before valid
  files in fallback paths

### Out

- V16 health/cache/speed endpoints
- single-file classification logic
- route path or response model changes

## Design

Create `run_batch_classify_pipeline(...)` with:

- `files`
- `max_workers`
- `logger`
- injectable classifier getters for tests

Return:

- `total`
- `success`
- `failed`
- `processing_time`
- `results`

Also extract `build_batch_classify_item(...)` so fine/coarse label projection
stays in one place.

## Risk Controls

- preserve unsupported-format item error payloads
- preserve V16 and V6 result shapes and confidence rounding
- preserve Prometheus batch metrics emission
- lock route-level delegation with an integration monkeypatch
- lock mixed invalid/valid ordering so fallback paths do not overwrite the
  wrong result slot

## Validation Plan

1. `python3 -m py_compile src/core/classification/batch_classify_pipeline.py src/core/classification/__init__.py src/api/v1/analyze.py tests/unit/test_batch_classify_pipeline.py tests/integration/test_analyze_batch_classify_pipeline.py`
2. `.venv311/bin/flake8 src/core/classification/batch_classify_pipeline.py src/core/classification/__init__.py src/api/v1/analyze.py tests/unit/test_batch_classify_pipeline.py tests/integration/test_analyze_batch_classify_pipeline.py`
3. `.venv311/bin/python -m pytest -q tests/unit/test_batch_classify_pipeline.py tests/integration/test_analyze_batch_classify_pipeline.py tests/unit/test_v16_classifier_endpoints.py tests/unit/test_analysis_batch_pipeline.py tests/integration/test_analyze_batch_pipeline.py tests/unit/test_analysis_error_handling.py tests/integration/test_analyze_error_handling.py tests/unit/test_analysis_ocr_pipeline.py tests/integration/test_analyze_ocr_pipeline.py tests/unit/test_analysis_result_envelope.py tests/integration/test_analyze_result_envelope.py tests/unit/test_analysis_preflight.py tests/unit/test_document_pipeline.py tests/test_api_integration.py`
