# Phase 3 Vector Migration Pending Candidates Extraction Development Plan

## Goal

Extract the shared pending-candidate collection logic from `src/api/v1/vectors.py` into a core helper while keeping the route-level patch surface unchanged.

## Scope

- Add a shared helper under `src/core/`
- Keep `src.api.v1.vectors._collect_vector_migration_pending_candidates` as a thin compatibility wrapper
- Cover both memory and qdrant branches
- Add focused unit tests for the helper and wrapper delegation

## Files

- `src/core/vector_migration_pending_candidates.py`
- `src/api/v1/vectors.py`
- `tests/unit/test_vector_migration_pending_candidates.py`
- `tests/unit/test_vectors_migration_pending_candidates_delegation.py`

## Risk Controls

- Preserve `_get_qdrant_store_or_none` and `_resolve_vector_migration_scan_limit` usage in `vectors.py`
- Preserve `VectorMigrationPendingItem` construction shape
- Keep existing route tests as the main behavior lock
