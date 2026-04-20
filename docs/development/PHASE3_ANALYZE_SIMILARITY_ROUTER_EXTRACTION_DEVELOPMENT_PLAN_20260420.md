# Phase 3 Analyze Similarity Router Extraction Development Plan

## Goal
- Split the live `/api/v1/analyze/similarity*` routes out of `src/api/v1/analyze.py`.
- Keep route behavior unchanged while further shrinking the main analyze route module.

## Scope
- Add `src/api/v1/analyze_similarity_router.py`
- Update `src/api/v1/analyze.py` to include the split router
- Update direct qdrant patch tests to target the new router module
- Add route ownership smoke coverage

## Constraints
- Preserve route paths and response models
- Preserve qdrant-backed behavior and existing query pipeline semantics
- Keep changes limited to similarity routes only

## Validation Plan
- `py_compile` on touched files
- `flake8` on touched files
- Focused `pytest` for similarity endpoints, qdrant patch coverage, and router ownership
