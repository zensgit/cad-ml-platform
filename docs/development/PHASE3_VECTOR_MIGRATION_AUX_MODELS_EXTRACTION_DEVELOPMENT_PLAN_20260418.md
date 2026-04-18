# Phase 3 Vector Migration Auxiliary Models Extraction Development Plan

## Goal
- consolidate the remaining migration auxiliary schemas duplicated inside `vectors.py`
- keep the route contract unchanged while reducing schema drift across migration endpoints

## Scope
- extend `src/api/v1/vector_migration_models.py` with summary, pending, plan, preview, and trends schemas
- switch `src/api/v1/vectors.py` to import those shared models instead of defining them inline
- expand shared schema smoke coverage in `tests/unit/test_vector_migration_models.py`

## Risk Controls
- preserve import compatibility from `src.api.v1.vectors`
- keep only schema ownership moving; route behavior and endpoint logic stay in place
- run preview, pending, plan, pending-run, status, and response-field regressions together

## Validation Plan
- `python3 -m py_compile ...`
- `.venv311/bin/flake8 ...`
- `.venv311/bin/python -m pytest -q` on shared model tests plus vectors migration regressions
