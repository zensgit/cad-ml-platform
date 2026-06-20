# CAD ML Vector Migration Pending Run Candidates Helper Ownership Development 20260518

## Goal

Continue Phase 3 helper ownership cleanup by moving vector migration pending-run
candidate collection call mapping out of the shared pending-run execution
pipeline.

This slice targets the small request-to-candidate-collector mapping step before
the pending-run guard and migrate-request construction run.

## Changes

- Added `src/core/vector_migration_pending_run_candidates.py` with
  `collect_pending_run_candidates`.
- Updated `src/core/vector_migration_pending_run_pipeline.py` so
  `run_vector_migration_pending_run_pipeline` delegates pending-candidate
  collection argument mapping to the new helper.
- Kept the public pending-run execution entrypoint unchanged:
  `src.core.vector_migration_pending_run_pipeline.run_vector_migration_pending_run_pipeline`.
- Kept route and facade behavior unchanged. The split write router still calls
  through `src.api.v1.vectors.run_vector_migration_pending_run_pipeline`.
- Added `tests/unit/test_vector_migration_pending_run_candidates.py` for:
  - mapping `payload.limit` to collector `limit`.
  - passing through `target_version`.
  - passing through `payload.from_version_filter`.
  - preserving `None` filters and returning the pending dict unchanged.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Compatibility Notes

- Runtime API behavior is unchanged.
- OpenAPI models and route registration were not touched.
- The public pending-run entrypoint remains
  `src.core.vector_migration_pending_run_pipeline.run_vector_migration_pending_run_pipeline`.
- The route-level monkeypatch surface remains
  `src.api.v1.vectors.run_vector_migration_pending_run_pipeline`.
- The helper does not normalize filters or mutate collector results. It preserves
  the existing behavior where filter normalization is owned by the downstream
  pending-candidate collector.

## Claude Code Assistance

Claude Code was used as a read-only compatibility reviewer.

Invocation boundary:

- stdin-only targeted snippets.
- no tools enabled.
- no file writes.
- no secrets or environment dumps.

Reviewer result: compatible. Claude Code returned OK for preserving `limit`,
`target_version`, `from_version_filter` including `None`, unchanged pending dict
return, and the public pending-run pipeline entrypoint.

## Remaining Work

- Continue remaining shared helper ownership cleanup only where a direct helper
  boundary can be covered by focused tests.
- Preserve `src.api.v1.vectors.*` facade wrappers and route-level monkeypatch
  surfaces during router closeout.
- Keep pending-run, pending-candidate, plan, summary, and delegation tests in
  the target validation set when changing vector migration execution behavior.
