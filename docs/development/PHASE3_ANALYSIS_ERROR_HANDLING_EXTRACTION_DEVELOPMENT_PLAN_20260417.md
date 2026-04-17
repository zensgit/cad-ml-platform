# Phase 3 Analysis Error Handling Extraction Development Plan

## Goal

Extract the analyze-route exception wrapping chain from `src/api/v1/analyze.py`
into a shared helper while preserving:

- options JSON parse error contract
- non-structured `HTTPException` wrapping behavior
- structured `HTTPException` passthrough behavior
- unexpected exception logging and `500` wrapping behavior
- existing metrics side effects for each error path

## Scope

### In

- add `src/core/analysis_error_handling.py`
- move JSON options parse error handling
- move `HTTPException` remapping / wrapping logic
- move unexpected exception logging / wrapping logic
- keep `analyze.py` as thin `except ...: handle_*()` calls

### Out

- successful result path
- business pipeline orchestration
- vector / OCR / document / classification logic
- route path or response schema changes

## Design

Create three helpers:

- `handle_analysis_options_json_error()`
- `handle_analysis_http_exception(exc)`
- `handle_analysis_unexpected_exception(file_name, exc, logger_instance)`

Each helper raises `HTTPException` directly after applying the same metric and
error-payload behavior currently implemented inline.

`analyze.py` keeps:

- route-level `try` scope
- simple `except` branches that delegate to the shared helpers

## Risk Controls

- preserve `JSON_PARSE_ERROR` payload shape and `400` status
- preserve `HTTPException` status-to-error-code mapping for `400/404/413/422`
- preserve `dict` detail passthrough for already-structured errors
- preserve unexpected exception log line content

## Validation Plan

1. `python3 -m py_compile src/core/analysis_error_handling.py src/api/v1/analyze.py tests/unit/test_analysis_error_handling.py tests/integration/test_analyze_error_handling.py`
2. `.venv311/bin/flake8 src/core/analysis_error_handling.py src/api/v1/analyze.py tests/unit/test_analysis_error_handling.py tests/integration/test_analyze_error_handling.py`
3. `.venv311/bin/python -m pytest -q tests/unit/test_analysis_error_handling.py tests/integration/test_analyze_error_handling.py tests/unit/test_error_codes_analysis.py tests/unit/test_analysis_ocr_pipeline.py tests/integration/test_analyze_ocr_pipeline.py tests/unit/test_analysis_result_envelope.py tests/integration/test_analyze_result_envelope.py tests/unit/test_analysis_preflight.py tests/unit/test_document_pipeline.py tests/test_api_integration.py`
