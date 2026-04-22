# Phase 3 Analyze Vector Update Router Extraction Development Plan

## Goal

Extract the legacy `/api/v1/analyze/vectors/update` route into a dedicated split router and retire the now-empty `analyze_vector_compat.py` module.

## Scope

- Add a dedicated analyze vector update router module
- Update `analyze.py` to include the new router
- Remove the obsolete `analyze_vector_compat.py`
- Update integration monkeypatch tests and route ownership smoke coverage

## Files

- `src/api/v1/analyze_vector_update_router.py`
- `src/api/v1/analyze.py`
- `tests/integration/test_analyze_vector_update_pipeline.py`
- `tests/unit/test_analyze_vector_update_router.py`

## Risk Controls

- Keep the route path unchanged: `/api/v1/analyze/vectors/update`
- Keep `run_vector_update_pipeline(...)` as the only business logic entry
- Validate route ownership and monkeypatch delegation against the new module
