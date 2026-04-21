# Phase 3 Vector Migrate Pipeline Extraction Development Plan

## Goal

Extract the main `/api/v1/vectors/migrate` execution flow from `src/api/v1/vectors.py` into a shared core helper while preserving route-level patch surfaces, history behavior, and metrics semantics.

## Scope

- Add a shared migrate pipeline under `src/core/`
- Keep `vectors.py` responsible for route signature, qdrant-store resolution, and history entry ownership
- Preserve monkeypatch compatibility for:
  - `src.api.v1.vectors._get_qdrant_store_or_none`
  - `src.core.feature_extractor.FeatureExtractor.upgrade_vector`
  - `src.core.similarity._VECTOR_STORE`
  - `src.core.similarity._VECTOR_META`

## Files

- `src/core/vector_migrate_pipeline.py`
- `src/api/v1/vectors.py`
- `tests/unit/test_vector_migrate_pipeline.py`
- `tests/unit/test_vectors_migrate_delegation.py`

## Risk Controls

- Keep route-level wrapper thin and explicit
- Pass module-local dependencies into the helper instead of importing mutable globals at helper import time
- Validate against API, metrics, history, downgrade, and dimension-mismatch regressions
