# Phase 3 Legacy Admin Pipeline Extraction Development Plan

## Goal

Extract the remaining small legacy admin helpers around:

- process rules audit
- Faiss rebuild

into a shared core helper while preserving:

- `/api/v1/process/rules/audit`
- `/api/v1/analyze/vectors/faiss/rebuild`

behavior and keeping route surfaces thin.

## Scope

### In

- add `src/core/legacy_admin_pipeline.py`
- move process rules audit assembly into a shared helper
- move Faiss rebuild orchestration into a shared helper
- make `src/api/v1/process.py` delegate to the shared helper
- make `src/api/v1/analyze.py` delegate to the shared helper
- add helper-level and route-level regression coverage

### Out

- process route generation endpoints
- vector migration/update endpoints
- route path or response schema changes

## Design

Create:

- `run_process_rules_audit_pipeline(...)`
- `run_faiss_rebuild_pipeline(...)`

For process rules audit, keep dependency injection explicit:

- `load_rules_fn`
- `rules_path`
- `path_exists`
- `file_opener`

This preserves existing route tests that patch `process.py` symbols while
centralizing the actual audit assembly.

For Faiss rebuild, keep only:

- backend gate
- store construction
- rebuild result mapping

## Risk Controls

- preserve process rules audit response fields and file-hash behavior
- preserve existing `process.py` metrics behavior by leaving metric increments in
  the route wrapper
- preserve Faiss rebuild skip behavior for non-faiss backends
- validate helper behavior directly
- validate legacy analyze Faiss route delegates to the shared helper

## Validation Plan

1. `python3 -m py_compile src/core/legacy_admin_pipeline.py src/api/v1/analyze.py src/api/v1/process.py tests/unit/test_legacy_admin_pipeline.py tests/integration/test_analyze_legacy_admin_pipeline.py`
2. `.venv311/bin/flake8 src/core/legacy_admin_pipeline.py src/api/v1/analyze.py src/api/v1/process.py tests/unit/test_legacy_admin_pipeline.py tests/integration/test_analyze_legacy_admin_pipeline.py`
3. `.venv311/bin/python -m pytest -q tests/unit/test_legacy_admin_pipeline.py tests/integration/test_analyze_legacy_admin_pipeline.py tests/unit/test_faiss_rebuild.py tests/unit/test_process_rules_audit_endpoint.py tests/unit/test_process_rules_audit_raw_param.py tests/unit/test_process_api_coverage.py tests/unit/test_analysis_error_handling.py tests/integration/test_analyze_error_handling.py tests/unit/test_analysis_ocr_pipeline.py tests/integration/test_analyze_ocr_pipeline.py tests/unit/test_analysis_result_envelope.py tests/integration/test_analyze_result_envelope.py tests/unit/test_analysis_preflight.py tests/unit/test_document_pipeline.py tests/test_api_integration.py`
