# Phase 3 Vector Delete Pipeline Extraction Development Plan

## Goal
- Extract the `/api/v1/vectors/delete` orchestration body out of `src/api/v1/vectors.py`.
- Keep the route signature and existing `src.api.v1.vectors.*` patch surface unchanged.

## Scope
- Add `src/core/vector_delete_pipeline.py`
- Update `src/api/v1/vectors.py` to delegate the delete body to the shared helper
- Add direct helper coverage
- Add a thin route delegation regression test

## Constraints
- Preserve the existing FastAPI route path and request/response contract
- Preserve qdrant and memory/faiss/redis delete behavior
- Preserve module-level patch points such as `_get_qdrant_store_or_none` and `get_client` by passing them into the helper at call time
- Keep changes limited to delete extraction only

## Validation Plan
- `py_compile` on touched files
- `flake8` on touched files
- Focused `pytest` for the new helper, route delegation, and existing vector delete regressions
