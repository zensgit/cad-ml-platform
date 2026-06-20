# CAD ML Vector Migration Pending Summary Helper Ownership Development 20260517

## Goal

Continue Phase 3 helper ownership cleanup by moving vector migration
pending-summary payload construction out of the shared plan payload module.

This slice targets summary-only behavior: ranked recommendations, largest
pending source version, complete-distribution pending ratio, partial-scan ratio
suppression, and empty-distribution output.

## Changes

- Added `src/core/vector_migration_pending_summary.py` with
  `build_vector_migration_pending_summary_payload`.
- Updated `src/core/vector_migration_plan_pipeline.py` so it imports and
  re-exports the pending-summary payload builder while preserving
  `build_vector_migration_plan_payload`.
- Kept the public summary payload entrypoint unchanged:
  `src.core.vector_migration_plan_pipeline.build_vector_migration_pending_summary_payload`.
- Kept route and facade behavior unchanged. The split routers still call through
  `src.api.v1.vectors.build_vector_migration_pending_summary_payload`.
- Added `tests/unit/test_vector_migration_pending_summary_helper.py` for:
  - complete-distribution summary payloads.
  - partial Qdrant scan ratio suppression.
  - empty-distribution behavior.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Compatibility Notes

- Runtime API behavior is unchanged.
- OpenAPI models and route registration were not touched.
- The public summary payload entrypoint remains
  `src.core.vector_migration_plan_pipeline.build_vector_migration_pending_summary_payload`.
- The route-level monkeypatch surface remains
  `src.api.v1.vectors.build_vector_migration_pending_summary_payload`.
- Ranking continues to use the shared count-descending and version-name
  ascending order from `src.core.vector_migration_plan_batches.rank_observed_versions`.
- Complete memory distributions still calculate:
  `round(total_pending / max(scanned_vectors, 1), 4)`.
- Partial Qdrant distributions still return `pending_ratio=None`.

## Claude Code Assistance

Claude Code was used as a read-only compatibility reviewer.

Invocation boundary:

- stdin-only targeted snippets.
- no tools enabled.
- no file writes.
- no secrets or environment dumps.

Reviewer result: compatible. Claude Code confirmed ranked recommendations,
largest pending version/count, complete-distribution pending ratio,
partial-scan ratio suppression, empty-distribution behavior, and the public
entrypoint through `vector_migration_plan_pipeline` were preserved.

## Remaining Work

- Continue remaining shared helper ownership cleanup only where a direct helper
  boundary can be covered by focused tests.
- Preserve `src.api.v1.vectors.*` facade wrappers and route-level monkeypatch
  surfaces during router closeout.
- Keep pending summary, plan, pending-run, and delegation tests in the target
  validation set when changing vector migration planning behavior.
