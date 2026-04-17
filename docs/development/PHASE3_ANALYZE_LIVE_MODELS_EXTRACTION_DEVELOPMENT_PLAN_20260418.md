# Phase 3 Analyze Live Models Extraction Development Plan

## Goal
- move the remaining live route schemas out of `src/api/v1/analyze.py`
- keep `analyze.py` focused on route wiring and helper orchestration
- preserve existing imports by re-exporting the schema names from `analyze.py`

## Scope
- add `src/api/v1/analyze_live_models.py`
- move the main analyze, batch classify, and similarity schemas there
- import those models back into `src/api/v1/analyze.py`
- add a small schema smoke test module

## Risk Controls
- do not change schema fields or defaults
- keep `AnalysisOptions` importable from `src.api.v1.analyze`
- run history-sequence, similarity, and batch-classify regressions

## Validation Plan
- `python3 -m py_compile ...`
- `.venv311/bin/flake8 ...`
- `.venv311/bin/python -m pytest -q` on live model tests plus existing analyze similarity/batch-classify regressions
