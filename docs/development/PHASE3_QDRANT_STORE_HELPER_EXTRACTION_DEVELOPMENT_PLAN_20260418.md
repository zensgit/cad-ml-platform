# Phase 3 Qdrant Store Helper Extraction Development Plan

## Goal
- remove duplicated Qdrant backend resolver logic across API route modules
- keep each route module patchable by preserving the local `_get_qdrant_store_or_none` alias

## Scope
- add shared helper `src/core/qdrant_store_helper.py`
- switch `analyze.py`, `compare.py`, `features.py`, `maintenance.py`, `vectors.py`, and `vectors_stats.py` to import the shared resolver
- add a focused helper unit test and rerun route regressions that patch those aliases

## Risk Controls
- keep the public local helper name in each route module via import alias
- move only the duplicated resolver body, not route behavior
- validate compare/features/vectors/maintenance/analyze qdrant-path regressions

## Validation Plan
- `python3 -m py_compile ...`
- `.venv311/bin/flake8 ...`
- `.venv311/bin/python -m pytest -q` on helper tests plus route regressions
