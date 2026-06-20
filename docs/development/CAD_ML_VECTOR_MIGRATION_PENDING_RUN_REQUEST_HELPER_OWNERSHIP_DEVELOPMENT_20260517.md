# CAD ML Vector Migration Pending Run Request Helper Ownership Development 20260517

## Goal

Continue Phase 3 helper ownership cleanup by moving vector migration pending-run
migrate-request construction out of the shared pending-run execution pipeline.

This slice targets the small request-shaping step that turns pending candidate
ids into the existing migration request object.

## Changes

- Added `src/core/vector_migration_pending_run_request.py` with
  `build_pending_run_migrate_request`.
- Updated `src/core/vector_migration_pending_run_pipeline.py` so
  `run_vector_migration_pending_run_pipeline` delegates request construction to
  the new helper before calling `migrate_vectors_fn`.
- Kept the public pending-run execution entrypoint unchanged:
  `src.core.vector_migration_pending_run_pipeline.run_vector_migration_pending_run_pipeline`.
- Kept route and facade behavior unchanged. The split write router still calls
  through `src.api.v1.vectors.run_vector_migration_pending_run_pipeline`.
- Added `tests/unit/test_vector_migration_pending_run_request.py` for:
  - mapping `pending_ids` to request `ids`.
  - mapping target version to `to_version`.
  - preserving `dry_run`.
  - preserving an empty pending id list.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Compatibility Notes

- Runtime API behavior is unchanged.
- OpenAPI models and route registration were not touched.
- The public pending-run entrypoint remains
  `src.core.vector_migration_pending_run_pipeline.run_vector_migration_pending_run_pipeline`.
- The route-level monkeypatch surface remains
  `src.api.v1.vectors.run_vector_migration_pending_run_pipeline`.
- `migrate_vectors_fn` is still called with the constructed request object and
  the same keyword-only `api_key`.
- The helper does not validate ids or mutate pending data; it preserves the
  existing pipeline behavior and delegates request class validation to
  `request_cls`.

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
