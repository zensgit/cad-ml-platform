# Phase 3 Analyze Result Router Extraction Development Plan

## Goal
- Split the live `GET /api/v1/analyze/{analysis_id}` route out of `src/api/v1/analyze.py`.
- Keep route behavior unchanged while further shrinking the main analyze route module.

## Scope
- Add `src/api/v1/analyze_result_router.py`
- Update `src/api/v1/analyze.py` to include the split result router
- Add route ownership smoke coverage
- Restore deprecated `/api/v1/analyze/vectors` 410 regression coverage now that the catch-all route is included last

## Constraints
- Preserve the catch-all route path and 404 behavior
- Keep the result route included after other analyze child routes
- Preserve cache-backed lookup behavior via `load_analysis_result_with_cache`
- Keep changes limited to the result route split and adjacent routing regressions

## Validation Plan
- `py_compile` on touched files
- `flake8` on touched files
- Focused `pytest` for route ownership, deprecated endpoint coverage, and direct result-route behavior
