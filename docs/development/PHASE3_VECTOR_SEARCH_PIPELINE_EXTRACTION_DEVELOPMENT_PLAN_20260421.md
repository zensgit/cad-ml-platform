# Phase 3 Vector Search Pipeline Extraction Development Plan

## Goal
- Extract the `/api/v1/vectors/search` orchestration body out of `src/api/v1/vectors.py`.
- Keep the route signature and existing `src.api.v1.vectors.*` patch surface unchanged.

## Scope
- Add `src/core/vector_search_pipeline.py`
- Update `src/api/v1/vectors.py` to delegate the search body to the shared helper
- Add direct helper coverage
- Add a thin route delegation regression test

## Constraints
- Preserve the existing FastAPI route path and request/response contract
- Preserve qdrant and memory search behavior
- Preserve the `src.api.v1.vectors._get_qdrant_store_or_none` patch point by passing it into the helper at call time
- Keep changes limited to search extraction only

## Validation Plan
- `py_compile` on touched files
- `flake8` on touched files
- Focused `pytest` for the new helper, route delegation, and existing vector search regressions
