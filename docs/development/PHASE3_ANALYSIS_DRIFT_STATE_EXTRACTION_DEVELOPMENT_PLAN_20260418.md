# Phase 3 Analysis Drift State Extraction Development Plan

## Goal
- remove the inline `_DRIFT_STATE` definition from `src/api/v1/analyze.py`
- preserve the existing `src.api.v1.analyze._DRIFT_STATE` compatibility surface used by drift routes, startup hooks, and tests

## Scope
- add shared state module `src/core/analysis_drift_state.py`
- switch `src/api/v1/analyze.py` to alias the shared state object
- add focused unit coverage for shared state shape, fresh-state creation, and analyze compatibility export

## Risk Controls
- keep the shared drift state object identity stable for `analyze.py`, `drift.py`, `main.py`, and tests that mutate `_DRIFT_STATE`
- preserve the current state keys and default values
- do not change drift route behavior, Redis persistence keys, or auto-refresh logic

## Validation Plan
- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile ...`
- `.venv311/bin/flake8 ...`
- `.venv311/bin/python -m pytest -q` on shared state smoke tests plus existing drift regressions
