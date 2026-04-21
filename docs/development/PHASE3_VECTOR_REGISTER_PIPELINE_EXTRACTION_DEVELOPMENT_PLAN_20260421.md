# Phase 3 Vector Register Pipeline Extraction Development Plan

## Goal
- Extract the `/api/v1/vectors/register` orchestration body out of `src/api/v1/vectors.py`.
- Keep the route signature and existing `src.api.v1.vectors.*` patch surface unchanged.

## Scope
- Add `src/core/vector_register_pipeline.py`
- Update `src/api/v1/vectors.py` to delegate the register body to the shared helper
- Add direct helper coverage
- Add a thin route delegation regression test

## Constraints
- Preserve the existing FastAPI route path and request/response contract
- Preserve qdrant and memory/faiss registration behavior
- Preserve the `src.api.v1.vectors._get_qdrant_store_or_none` patch point by passing it into the helper at call time
- Keep changes limited to register extraction only

## Validation Plan
- `py_compile` on touched files
- `flake8` on touched files
- Focused `pytest` for the new helper, route delegation, and existing vector register regressions
