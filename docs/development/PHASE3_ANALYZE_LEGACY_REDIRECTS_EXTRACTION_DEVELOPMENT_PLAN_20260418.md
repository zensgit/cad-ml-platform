# Phase 3 Analyze Legacy Redirects Extraction Development Plan

## Goal
- extract repeated legacy 410 redirect behavior out of `src/api/v1/analyze.py`
- make the deprecated analyze auxiliary endpoints delegate to one shared helper
- keep response shape and migration metadata unchanged

## Scope
- add `src/core/legacy_redirect_pipeline.py`
- update the deprecated analyze endpoints that now only raise 410 migration errors
- add helper tests and route delegation regression tests

## Risk Controls
- preserve `create_migration_error()` payload format
- keep paths, methods, and response models unchanged
- run existing deprecated endpoint regressions in addition to new helper tests

## Validation Plan
- `python3 -m py_compile ...`
- `.venv311/bin/flake8 ...`
- `.venv311/bin/python -m pytest -q` on the helper, route delegation, and existing deprecated endpoint tests
