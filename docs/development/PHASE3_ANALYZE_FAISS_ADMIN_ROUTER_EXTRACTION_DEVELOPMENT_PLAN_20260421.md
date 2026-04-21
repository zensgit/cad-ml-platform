# Phase 3 Analyze Faiss Admin Router Extraction Development Plan

## Goal

Extract the legacy `/api/v1/analyze/vectors/faiss/rebuild` route into its own split router while preserving the original path, response shape, and shared pipeline behavior.

## Scope

- Add a dedicated analyze faiss admin router module
- Update `analyze.py` to include the new router
- Remove the `faiss rebuild` endpoint from `analyze_vector_compat.py`
- Update ownership and integration patch tests

## Files

- `src/api/v1/analyze_faiss_admin_router.py`
- `src/api/v1/analyze.py`
- `src/api/v1/analyze_vector_compat.py`
- `tests/integration/test_analyze_legacy_admin_pipeline.py`
- `tests/unit/test_analyze_vector_compat_router.py`
- `tests/unit/test_analyze_faiss_admin_router.py`

## Risk Controls

- Keep the route path unchanged: `/api/v1/analyze/vectors/faiss/rebuild`
- Keep `run_faiss_rebuild_pipeline(...)` as the only business logic entry
- Validate both route ownership and monkeypatch-based delegation
