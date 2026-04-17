# Phase 3 Similarity Query Pipeline Extraction Development Plan

## Goal

Extract the `/api/v1/analyze/similarity` and `/api/v1/analyze/similarity/topk` read-side
dispatch logic from `src/api/v1/analyze.py` into a shared helper while keeping:

- route paths unchanged
- request/response schemas unchanged
- memory / Faiss / Qdrant behavior unchanged
- error semantics unchanged

## Scope

### In

- add `src/core/vector_query_pipeline.py`
- move pairwise similarity query dispatch into the helper
- move top-k similarity dispatch into the helper
- keep `analyze.py` as a thin FastAPI adapter
- add direct unit coverage for the helper

### Out

- vector admin endpoints
- batch similarity in `src/api/v1/vectors.py`
- route path changes
- response model changes

## Design

### New helper

Create `src/core/vector_query_pipeline.py` with:

- `run_similarity_query_pipeline(...)`
- `run_similarity_topk_pipeline(...)`
- `matches_similarity_topk_filters(...)`

The helper owns:

- Qdrant vs in-memory / Faiss branching
- not-found and dimension-mismatch result shaping
- label contract projection
- top-k filter condition assembly
- memory-side top-k filter matching

### Route changes

Keep request and response models in `src/api/v1/analyze.py`, but replace the heavy route
bodies with:

- helper invocation
- response model construction
- existing metric wiring via callbacks

## Risk Controls

- preserve `analysis_error_code_total` increments through callback injection
- preserve `vector_query_latency_seconds` recording through callback injection
- keep Qdrant exception swallow/fallback behavior for top-k
- validate against the existing similarity endpoint regression suite

## Validation Plan

Run:

1. `python3 -m py_compile src/core/vector_query_pipeline.py src/api/v1/analyze.py tests/unit/test_vector_query_pipeline.py`
2. `.venv311/bin/flake8 src/core/vector_query_pipeline.py src/api/v1/analyze.py tests/unit/test_vector_query_pipeline.py`
3. `.venv311/bin/python -m pytest -q tests/unit/test_vector_query_pipeline.py tests/unit/test_similarity_endpoint.py tests/unit/test_similarity_topk.py tests/unit/test_similarity_filters.py tests/unit/test_similarity_topk_pagination.py tests/unit/test_similarity_error_codes.py tests/unit/test_similarity_complexity_filter.py`
