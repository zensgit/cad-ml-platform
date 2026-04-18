# Phase 3 Vector Migration Models Extraction Development Plan

## Goal
- consolidate duplicated vector migration schemas shared by `analyze.py` legacy routes and `vectors.py`
- keep route behavior unchanged while reducing schema drift risk

## Scope
- add `src/api/v1/vector_migration_models.py`
- switch `src/api/v1/vectors.py` to import the shared migration request/response schemas
- switch `src/api/v1/analyze_aux_models.py` to re-export the same shared schemas
- add a small shared schema smoke test

## Risk Controls
- use the richer `vectors.py` status schema as the shared contract so legacy routes can safely return subsets
- preserve import compatibility from both route modules
- run vectors migration and legacy analyze migration regressions

## Validation Plan
- `python3 -m py_compile ...`
- `.venv311/bin/flake8 ...`
- `.venv311/bin/python -m pytest -q` on shared model tests plus vectors/analyze migration regressions
