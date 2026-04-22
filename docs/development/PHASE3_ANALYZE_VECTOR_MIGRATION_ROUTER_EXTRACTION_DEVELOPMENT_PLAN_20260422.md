# Phase 3 Analyze Vector Migration Router Extraction Development Plan

## Goal

Extract the legacy `/api/v1/analyze/vectors/migrate*` routes into a dedicated split router while preserving the existing paths, response models, and shared migration pipeline behavior.

## Scope

- Add a dedicated analyze vector migration router module
- Update `analyze.py` to include the new router
- Remove `migrate` and `migrate/status` from `analyze_vector_compat.py`
- Update route ownership and integration delegation tests

## Files

- `src/api/v1/analyze_vector_migration_router.py`
- `src/api/v1/analyze.py`
- `src/api/v1/analyze_vector_compat.py`
- `tests/integration/test_analyze_legacy_vector_migration_pipeline.py`
- `tests/unit/test_analyze_vector_compat_router.py`
- `tests/unit/test_analyze_vector_migration_router.py`

## Risk Controls

- Keep route paths unchanged:
  - `/api/v1/analyze/vectors/migrate`
  - `/api/v1/analyze/vectors/migrate/status`
- Keep shared legacy migration pipeline functions as the only business logic entry
- Validate both route ownership and monkeypatch-based delegation
