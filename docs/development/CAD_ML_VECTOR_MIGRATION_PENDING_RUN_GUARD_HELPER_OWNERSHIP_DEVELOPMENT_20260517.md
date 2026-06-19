# CAD ML Vector Migration Pending Run Guard Helper Ownership Development 20260517

## Goal

Continue Phase 3 helper ownership cleanup by moving vector migration pending-run
partial-scan guard logic out of the shared pending-run execution pipeline.

This slice targets the Qdrant partial-scan safety check before executing a
pending migration run.

## Changes

- Added `src/core/vector_migration_pending_run_guard.py` with
  `ensure_pending_run_scan_is_allowed`.
- Updated `src/core/vector_migration_pending_run_pipeline.py` so
  `run_vector_migration_pending_run_pipeline` delegates the partial-scan guard
  to the new helper.
- Kept the public pending-run execution entrypoint unchanged:
  `src.core.vector_migration_pending_run_pipeline.run_vector_migration_pending_run_pipeline`.
- Kept route and facade behavior unchanged. The split write router still calls
  through `src.api.v1.vectors.run_vector_migration_pending_run_pipeline`.
- Added `tests/unit/test_vector_migration_pending_run_guard.py` for:
  - complete Qdrant scans.
  - partial Qdrant scans with explicit override.
  - memory scans.
  - partial Qdrant scans without override producing the existing 409 error
    structure.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Compatibility Notes

- Runtime API behavior is unchanged.
- OpenAPI models and route registration were not touched.
- The public pending-run entrypoint remains
  `src.core.vector_migration_pending_run_pipeline.run_vector_migration_pending_run_pipeline`.
- The route-level monkeypatch surface remains
  `src.api.v1.vectors.run_vector_migration_pending_run_pipeline`.
- The 409 error keeps:
  - `status_code=409`.
  - `code=CONSTRAINT_VIOLATION`.
  - `stage=vector_migrate_pending_run`.
  - context fields for `target_version`, `scanned_vectors`, and `scan_limit`.
- Complete Qdrant scans, memory scans, and partial Qdrant scans with
  `allow_partial_scan=true` remain allowed.

## Claude Code Assistance

Claude Code was attempted as a read-only compatibility reviewer.

Invocation boundary:

- stdin-only targeted snippets.
- no tools enabled.
- no file writes.
- no secrets or environment dumps.

Reviewer result: blocked by local Claude Code quota. The CLI returned:
`You've hit your limit` with a reset time of `10:30pm (America/Los_Angeles)`.

## Remaining Work

- Continue remaining shared helper ownership cleanup only where a direct helper
  boundary can be covered by focused tests.
- Preserve `src.api.v1.vectors.*` facade wrappers and route-level monkeypatch
  surfaces during router closeout.
- Re-run the Claude Code read-only review after quota reset if an external
  reviewer signal is required for this slice.
