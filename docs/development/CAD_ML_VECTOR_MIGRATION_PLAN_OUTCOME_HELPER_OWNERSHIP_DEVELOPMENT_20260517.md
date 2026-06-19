# CAD ML Vector Migration Plan Outcome Helper Ownership Development 20260517

## Goal

Continue Phase 3 helper ownership cleanup by moving vector migration plan outcome
metrics out of the shared plan payload module.

This slice targets post-batch planning behavior: blocking reasons, first
recommended request, planned and remaining counts, planned ratio, coverage
completion, truncation flags, and suggested next batch count.

## Changes

- Added `src/core/vector_migration_plan_outcome.py` with
  `build_vector_migration_plan_outcome`.
- Updated `src/core/vector_migration_plan_pipeline.py` so
  `build_vector_migration_plan_payload` delegates post-batch outcome metrics to
  the new helper.
- Kept the public plan payload entrypoint unchanged:
  `src.core.vector_migration_plan_pipeline.build_vector_migration_plan_payload`.
- Kept route and facade behavior unchanged. The split routers still call through
  `src.api.v1.vectors.build_vector_migration_plan_payload`.
- Added `tests/unit/test_vector_migration_plan_outcome.py` for:
  - complete but truncated plan coverage.
  - partial-scan blocking behavior.
  - no-batch blocking behavior.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Compatibility Notes

- Runtime API behavior is unchanged.
- OpenAPI models and route registration were not touched.
- The public plan payload entrypoint remains
  `src.core.vector_migration_plan_pipeline.build_vector_migration_plan_payload`.
- The route-level monkeypatch surface remains
  `src.api.v1.vectors.build_vector_migration_plan_payload`.
- Blocking reasons keep their existing order:
  `partial_scan_override_required` before `no_pending_vectors`.
- `recommended_first_batch` remains the first batch object when batches exist,
  and `recommended_first_request_payload` remains a copied request payload.
- Remaining counts and planned ratios are still only calculated for complete
  distributions with known `total_pending`.

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
