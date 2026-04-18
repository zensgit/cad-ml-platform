# Phase 3 Analyze Legacy Redirect Router Extraction Development Plan

## Goal
- remove the cluster of pure legacy redirect routes from `src/api/v1/analyze.py`
- preserve the existing `/api/v1/analyze/...` deprecated endpoints and their migration metadata unchanged

## Scope
- add split router module `src/api/v1/analyze_legacy_redirects.py`
- switch `src/api/v1/analyze.py` to include the split router
- update focused integration tests that patch the legacy redirect pipeline hook
- add a small route ownership smoke test

## Risk Controls
- keep all legacy paths and HTTP methods unchanged
- keep `raise_legacy_redirect(...)` semantics unchanged
- preserve current 410 structured error payloads and migration metadata
- ensure the split router remains mounted under the existing `/api/v1/analyze` prefix

## Validation Plan
- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile ...`
- `.venv311/bin/flake8 ...`
- `.venv311/bin/python -m pytest -q` on deprecated endpoint, legacy redirect pipeline, and route ownership regressions
