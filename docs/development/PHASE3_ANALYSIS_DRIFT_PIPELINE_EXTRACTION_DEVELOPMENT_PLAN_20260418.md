# Phase 3 Analysis Drift Pipeline Extraction Development Plan

## Goal
- remove the inline drift recording block from `src/api/v1/analyze.py`
- keep the existing drift state shape, cache persistence keys, and metric observers unchanged

## Scope
- add shared helper `src/core/analysis_drift_pipeline.py`
- switch `src/api/v1/analyze.py` to call the shared helper for per-analysis drift updates
- add focused helper unit tests for baseline creation and prediction fallback behavior

## Risk Controls
- preserve the existing `_DRIFT_STATE` dictionary contract because drift routes and tests depend on it
- preserve baseline persistence keys: `baseline:material`, `baseline:material:ts`, `baseline:class`, `baseline:class:ts`
- preserve current analyze semantics, including `type -> ml_predicted_type` fallback and not mutating local baseline timestamps

## Validation Plan
- `python3 -m py_compile ...`
- `.venv311/bin/flake8 ...`
- `.venv311/bin/python -m pytest -q` on helper tests plus drift route/state regressions
