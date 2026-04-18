# Phase 3 Qdrant Similarity Helper Extraction Development Plan

## Goal
- remove the last local Qdrant cosine similarity helper body from `analyze.py`
- keep the vector pipeline call contract unchanged by passing a shared helper callable

## Scope
- add shared helper `src/core/qdrant_similarity_helper.py`
- switch `src/api/v1/analyze.py` to import and pass the shared helper into `run_vector_pipeline`
- add focused helper unit coverage and rerun vector pipeline analyze regressions

## Risk Controls
- do not touch the `_shadow_pipeline` compatibility aliases that tests still import from `analyze.py`
- move only the Qdrant similarity body, not vector registration or route behavior
- validate both the new helper payloads and the existing analyze/vector pipeline integration hook

## Validation Plan
- `python3 -m py_compile ...`
- `.venv311/bin/flake8 ...`
- `.venv311/bin/python -m pytest -q` on helper tests plus analyze/vector pipeline regressions
