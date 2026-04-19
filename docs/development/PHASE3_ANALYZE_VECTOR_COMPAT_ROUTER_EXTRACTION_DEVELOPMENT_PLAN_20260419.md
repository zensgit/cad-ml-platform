# Phase 3 Analyze Vector Compat Router Extraction Development Plan

## Goal
- remove the legacy vector admin/update thin wrappers from `src/api/v1/analyze.py`
- preserve the existing `/api/v1/analyze/vectors/...` compatibility endpoints unchanged

## Scope
- add split router module `src/api/v1/analyze_vector_compat.py`
- switch `src/api/v1/analyze.py` to include the split router
- update focused integration tests that patch the vector compat pipeline hooks
- add a small route ownership smoke test

## Risk Controls
- keep all legacy analyze vector paths and HTTP methods unchanged
- keep the existing shared helper ownership unchanged:
  - `run_faiss_rebuild_pipeline`
  - `run_vector_update_pipeline`
  - `run_legacy_vector_migrate_pipeline`
  - `run_legacy_vector_migration_status_pipeline`
- preserve current response models and backward-compatible route semantics

## Validation Plan
- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile ...`
- `.venv311/bin/flake8 ...`
- `.venv311/bin/python -m pytest -q` on vector compat route ownership, integration delegation, and existing analyze vector endpoint regressions
