# Phase 3 Analysis Batch Pipeline Extraction Development Plan

## Goal

Extract the `/api/v1/analyze/batch` route loop into a shared helper while
preserving:

- per-file delegation to `analyze_cad_file`
- success / failure aggregation contract
- per-item error payload shape
- route response payload fields: `total`, `successful`, `failed`, `results`

## Scope

### In

- add `src/core/analysis_batch_pipeline.py`
- move per-file loop and result aggregation into a shared helper
- normalize successful Pydantic results via `model_dump()`
- call `analyze_cad_file` with explicit keyword arguments instead of fragile
  positional arguments
- keep `batch_analyze` as a thin caller

### Out

- single-file analyze business logic
- batch-classify endpoint
- route path or response schema changes

## Design

Create `run_batch_analysis(...)` with:

- `files`
- `options`
- `api_key`
- injectable `analyze_file_fn`

Return:

- `total`
- `successful`
- `failed`
- `results`

`batch_analyze` keeps:

- route signature and auth dependency
- one helper call

## Risk Controls

- preserve item-level `{"file_name": ..., "error": ...}` failure payloads
- preserve success count based on absence of `"error"`
- validate route-level use of the shared helper via integration monkeypatch
- lock explicit `api_key=` keyword delegation in helper-level tests

## Validation Plan

1. `python3 -m py_compile src/core/analysis_batch_pipeline.py src/api/v1/analyze.py tests/unit/test_analysis_batch_pipeline.py tests/integration/test_analyze_batch_pipeline.py`
2. `.venv311/bin/flake8 src/core/analysis_batch_pipeline.py src/api/v1/analyze.py tests/unit/test_analysis_batch_pipeline.py tests/integration/test_analyze_batch_pipeline.py`
3. `.venv311/bin/python -m pytest -q tests/unit/test_analysis_batch_pipeline.py tests/integration/test_analyze_batch_pipeline.py tests/unit/test_analysis_error_handling.py tests/integration/test_analyze_error_handling.py tests/unit/test_analysis_ocr_pipeline.py tests/integration/test_analyze_ocr_pipeline.py tests/unit/test_analysis_result_envelope.py tests/integration/test_analyze_result_envelope.py tests/unit/test_analysis_preflight.py tests/unit/test_document_pipeline.py tests/test_api_integration.py`
