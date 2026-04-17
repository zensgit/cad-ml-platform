# Phase 3 Analysis Result Envelope Extraction Development Plan

## Goal

Extract the successful result-envelope block from `src/api/v1/analyze.py` into a
shared helper while preserving:

- `results["statistics"]` contract
- persistence to analysis cache and result store
- `analysis.completed` logging payload
- `processing_time` calculation
- final `AnalysisResult` response shape, including `cad_document`

## Scope

### In

- add `src/core/analysis_result_envelope.py`
- move statistics assembly into a shared helper
- move cad-document response serialization into a shared helper
- move persistence, success metric increment, and completion log
- keep `analyze.py` as a thin caller that returns `AnalysisResult(**payload)`

### Out

- request preflight / cache lookup
- document pipeline
- feature / classification / quality / process / vector flows
- error handling branches

## Design

Create `finalize_analysis_success(...)` with:

- request/result identifiers and timestamps
- `results`, `doc`, and `stage_times`
- persistence callbacks overrideable for unit tests
- logger / vector / material / unified-data context for completion logging

Return a response payload dict containing:

- `id`
- `timestamp`
- `file_name`
- `file_format`
- `results`
- `processing_time`
- `cache_hit`
- `cad_document`
- `feature_version`

`analyze.py` keeps:

- upstream business orchestration
- one call to `finalize_analysis_success(...)`
- route-level `AnalysisResult(**payload)` construction

## Risk Controls

- preserve 200-entity truncation in `cad_document`
- preserve statistics fields and stage timing passthrough
- preserve cache/store side effects and `analysis_result:{analysis_id}` key
- validate route-level use of the shared helper via integration monkeypatch

## Validation Plan

1. `python3 -m py_compile src/core/analysis_result_envelope.py src/api/v1/analyze.py tests/unit/test_analysis_result_envelope.py tests/integration/test_analyze_result_envelope.py`
2. `.venv311/bin/flake8 src/core/analysis_result_envelope.py src/api/v1/analyze.py tests/unit/test_analysis_result_envelope.py tests/integration/test_analyze_result_envelope.py`
3. `.venv311/bin/python -m pytest -q tests/unit/test_analysis_result_envelope.py tests/integration/test_analyze_result_envelope.py tests/unit/test_analysis_preflight.py tests/unit/test_document_pipeline.py tests/integration/test_analyze_vector_pipeline.py tests/test_api_integration.py`
