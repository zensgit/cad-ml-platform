# Phase 3 Vector Update Pipeline Extraction Development Plan

## Goal

Extract vector update orchestration into a shared helper so both:

- `/api/v1/vectors/update`
- `/api/v1/analyze/vectors/update`

delegate to the same implementation while preserving:

- memory backend behavior
- Qdrant backend behavior
- dimension mismatch handling
- structured error payloads
- update metadata behavior

## Scope

### In

- add `src/core/vector_update_pipeline.py`
- move shared update logic for memory and Qdrant backends into the helper
- keep both routes as thin delegates
- add direct helper tests and legacy analyze-route delegation coverage

### Out

- vector delete/list/search endpoints
- vector migration endpoints
- route paths or response model changes

## Design

Create `run_vector_update_pipeline(...)` with:

- `payload`
- optional `qdrant_store`

Return:

- `id`
- `status`
- `dimension`
- `error`
- `feature_version`

Use helper-private functions for:

- replace / append update application
- metadata merge
- non-enforced dimension mismatch response assembly

## Risk Controls

- preserve `409` conflict behavior when dimension enforcement is enabled
- preserve `dimension_mismatch` response shape when enforcement is disabled
- preserve Qdrant metadata update behavior
- validate the legacy analyze route only delegates, rather than retaining its own
  branchy implementation
- rely on existing `/api/v1/vectors/update` route tests to catch drift in the
  primary surface

## Validation Plan

1. `python3 -m py_compile src/core/vector_update_pipeline.py src/api/v1/analyze.py src/api/v1/vectors.py tests/unit/test_vector_update_pipeline.py tests/integration/test_analyze_vector_update_pipeline.py`
2. `.venv311/bin/flake8 src/core/vector_update_pipeline.py src/api/v1/analyze.py src/api/v1/vectors.py tests/unit/test_vector_update_pipeline.py tests/integration/test_analyze_vector_update_pipeline.py`
3. `.venv311/bin/python -m pytest -q tests/unit/test_vector_update_pipeline.py tests/integration/test_analyze_vector_update_pipeline.py tests/unit/test_vector_update.py tests/unit/test_vector_update_dimension_conflict.py tests/unit/test_vectors_module_endpoints.py tests/unit/test_analysis_error_handling.py tests/integration/test_analyze_error_handling.py tests/unit/test_analysis_ocr_pipeline.py tests/integration/test_analyze_ocr_pipeline.py tests/unit/test_analysis_result_envelope.py tests/integration/test_analyze_result_envelope.py tests/unit/test_analysis_preflight.py tests/unit/test_document_pipeline.py tests/test_api_integration.py`
