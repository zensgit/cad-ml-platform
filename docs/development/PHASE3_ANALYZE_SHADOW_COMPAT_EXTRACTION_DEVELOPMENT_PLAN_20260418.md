# Phase 3 Analyze Shadow Compat Extraction Development Plan

## Goal
- remove the local `shadow_pipeline` alias block from `src/api/v1/analyze.py`
- preserve the existing `src.api.v1.analyze` export surface for helper-level tests and patch points

## Scope
- add shared compat module `src/api/v1/analyze_shadow_compat.py`
- switch `src/api/v1/analyze.py` to import the four shadow helpers from the compat module
- add a focused smoke test to lock the re-export identities

## Risk Controls
- keep `_build_graph2d_soft_override_suggestion`, `_enrich_graph2d_prediction`, `_graph2d_is_drawing_type`, and `_resolve_history_sequence_file_path` importable from `src.api.v1.analyze`
- do not change helper behavior; only move the compatibility export surface
- preserve current implementation ownership in `src/core/classification/shadow_pipeline.py`

## Validation Plan
- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile ...`
- `.venv311/bin/flake8 ...`
- `.venv311/bin/python -m pytest -q` on compat smoke tests plus existing analyze/shadow regressions
