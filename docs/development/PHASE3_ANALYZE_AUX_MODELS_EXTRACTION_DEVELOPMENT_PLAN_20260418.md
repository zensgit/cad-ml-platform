# Phase 3 Analyze Auxiliary Models Extraction Development Plan

## Goal
- move the auxiliary and legacy endpoint response/request models out of `src/api/v1/analyze.py`
- reuse the shared `ProcessRulesAuditResponse` schema between `analyze.py` and `process.py`
- reduce route-file size without changing endpoint behavior

## Scope
- add `src/api/v1/analyze_aux_models.py`
- import the auxiliary/legacy schemas from that module in `analyze.py`
- switch `process.py` to reuse the shared `ProcessRulesAuditResponse`
- add a small schema smoke test module

## Risk Controls
- keep field names, defaults, and model config unchanged
- avoid unifying models whose schemas differ across primary/legacy routes
- run existing deprecated endpoint and process audit regressions

## Validation Plan
- `python3 -m py_compile ...`
- `.venv311/bin/flake8 ...`
- `.venv311/bin/python -m pytest -q` on the new model tests and existing process/deprecated endpoint suites
