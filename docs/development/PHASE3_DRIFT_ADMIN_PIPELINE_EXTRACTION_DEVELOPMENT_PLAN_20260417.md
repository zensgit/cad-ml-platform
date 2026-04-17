# Phase 3 Drift Admin Pipeline Extraction Development Plan

## Goal
- extract the duplicated drift admin logic out of route files
- make `src/api/v1/drift.py` a thin wrapper around a shared helper
- remove the dead legacy drift implementation still sitting in `src/api/v1/analyze.py`

## Scope
- add `src/core/drift_admin_pipeline.py`
- route `drift_status`, `drift_reset`, and `drift_baseline_status` through the helper
- preserve coarse prediction drift behavior in `src/api/v1/drift.py`
- remove the unused duplicate drift code from `src/api/v1/analyze.py`
- add helper tests and route delegation tests

## Risk Controls
- keep export/import baseline routes untouched in this slice
- preserve existing response models and route paths
- run existing drift regression tests alongside new helper tests

## Validation Plan
- `python3 -m py_compile ...`
- `.venv311/bin/flake8 ...`
- `.venv311/bin/python -m pytest -q` on the drift helper, route delegation, and existing drift regressions
