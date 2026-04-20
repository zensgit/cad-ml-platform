# Phase 3 Vector Batch Similarity Extraction Development Plan

## Goal
- Extract the `/api/v1/vectors/similarity/batch` orchestration body out of `src/api/v1/vectors.py`.
- Keep the route signature and existing `src.api.v1.vectors.*` patch surface unchanged.

## Scope
- Add `src/core/vector_batch_similarity.py`
- Update `src/api/v1/vectors.py` to delegate the batch similarity body to the shared helper
- Add direct helper coverage
- Add a thin route delegation regression test

## Constraints
- Preserve the existing FastAPI route path and request/response contract
- Preserve runtime behavior for both qdrant and memory/fallback branches
- Preserve the `src.api.v1.vectors._get_qdrant_store_or_none` patch point by passing it into the helper at call time
- Keep changes limited to batch similarity extraction only

## Validation Plan
- `py_compile` on touched files
- `flake8` on touched files
- Focused `pytest` for the new helper, route delegation, and existing batch similarity regressions
