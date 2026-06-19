# CAD ML Vector Migration Plan Batch Helper Ownership Development 20260517

## Goal

Continue Phase 3 helper ownership cleanup by moving vector migration plan batch
logic out of the shared plan payload module.

This slice targets pure plan construction behavior: observed-version ranking,
batch selection, suggested run limits, partial-scan override request payloads,
and run-count estimation.

## Changes

- Added `src/core/vector_migration_plan_batches.py` with:
  - `rank_observed_versions`.
  - `build_vector_migration_plan_batches`.
  - `estimate_migration_runs_by_version`.
- Updated `src/core/vector_migration_plan_pipeline.py` so
  `build_vector_migration_pending_summary_payload` and
  `build_vector_migration_plan_payload` delegate ranking, batch construction,
  and run estimation to the new helper module.
- Kept the public payload entrypoints unchanged:
  - `build_vector_migration_pending_summary_payload`.
  - `build_vector_migration_plan_payload`.
- Kept route and facade behavior unchanged. The split routers still call through
  `src.api.v1.vectors.build_vector_migration_plan_payload`.
- Added `tests/unit/test_vector_migration_plan_batches.py` for:
  - ranking by pending count descending and version name ascending.
  - `max_batches` truncation.
  - split-batch vs single-batch notes.
  - partial-scan override request payloads.
  - ceiling-based estimated run counts.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Compatibility Notes

- Runtime API behavior is unchanged.
- OpenAPI models and route registration were not touched.
- The public plan payload entrypoint remains
  `src.core.vector_migration_plan_pipeline.build_vector_migration_plan_payload`.
- The route-level monkeypatch surface remains
  `src.api.v1.vectors.build_vector_migration_plan_payload`.
- `recommended_from_versions` remains the full ranked version list, while
  `batches` remains truncated by `max_batches`.
- `estimated_runs_by_version` remains calculated from all observed versions,
  not just planned batches.
- Partial Qdrant scans still set:
  - `blocking_reasons=["partial_scan_override_required"]`.
  - `request_payload["allow_partial_scan"]=True`.
  - `notes` containing `partial_scan_override_required`.

## Claude Code Assistance

Claude Code was used as a read-only compatibility reviewer.

Invocation boundary:

- stdin-only targeted snippets.
- no tools enabled.
- no file writes.
- no secrets or environment dumps.

First invocation exceeded the low budget before returning review content. A
second smaller invocation completed successfully.

Reviewer result: compatible. Claude Code confirmed rank ordering, `max_batches`
truncation, split/single notes, partial-scan override propagation, estimated run
ceiling, and public plan payload entrypoint compatibility were preserved.

## Remaining Work

- Continue remaining shared helper ownership cleanup only when the behavior has
  a clear module boundary and direct tests.
- Preserve `src.api.v1.vectors.*` facade wrappers and route-level monkeypatch
  surfaces during any router closeout work.
- Keep pending summary, plan, pending-run, and delegation tests in the target
  validation set when changing vector migration planning behavior.
