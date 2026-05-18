# CAD ML Vector Migration Pending Qdrant Helper Ownership Development 20260515

## Goal

Continue Phase 3 helper ownership cleanup by moving the Qdrant branch of pending
candidate collection out of the shared memory/Qdrant entrypoint.

This slice targets Qdrant pending candidate scanning, including partial-scan
behavior, version filtering, pending-id collection, and item construction for
migration planning.

## Changes

- Added `src/core/vector_migration_pending_qdrant.py` with
  `collect_qdrant_migration_pending_candidates`.
- Updated `src/core/vector_migration_pending_candidates.py` so
  `collect_vector_migration_pending_candidates` delegates its Qdrant branch to
  the new helper.
- Kept the public shared entrypoint unchanged:
  `collect_vector_migration_pending_candidates`.
- Kept the API facade wrapper unchanged:
  `src.api.v1.vectors._collect_vector_migration_pending_candidates`.
- Added `tests/unit/test_vector_migration_pending_qdrant.py` for:
  - exact-count Qdrant scans.
  - page-limit behavior for `items` and returned `pending_ids`.
  - normalized filter application.
  - `unknown` feature-version handling for empty or missing metadata.
  - partial-scan semantics where `total_pending` is hidden.
- Extended `tests/unit/test_vector_migration_pending_candidates.py` to prove the
  shared entrypoint delegates Qdrant requests with normalized filters and the
  expected arguments.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Compatibility Notes

- Runtime API behavior is unchanged.
- Existing code that patches or calls
  `src.api.v1.vectors._collect_vector_migration_pending_candidates` remains
  supported.
- `collect_vector_migration_pending_candidates` still owns
  `from_version_filter` normalization before entering either backend branch.
- The Qdrant helper intentionally accepts `normalized_filter`; direct callers
  should pass `None` for no filter and use the shared entrypoint when accepting
  raw user input.
- The memory branch remains in the shared entrypoint and was not changed.

## Claude Code Assistance

Claude Code was used as a read-only compatibility reviewer.

Invocation boundary:

- stdin-only targeted snippets.
- no tools enabled.
- no file writes.
- no secrets or environment dumps.
- limited budget.

Reviewer result: compatible. Claude Code confirmed loop control, partial-scan
semantics, filter normalization boundary, `pending_ids[:limit]`, item capping,
and API facade delegation were preserved.

## Remaining Work

- Continue helper ownership cleanup only where a backend branch or helper has
  clear ownership and focused tests.
- Preserve `src.api.v1.vectors.*` facade wrappers while split routers still use
  them as monkeypatch surfaces.
- Keep migration pending/plan/run tests in the target validation set when
  changing pending-candidate behavior.
