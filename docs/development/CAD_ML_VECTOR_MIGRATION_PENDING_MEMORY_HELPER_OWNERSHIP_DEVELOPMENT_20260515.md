# CAD ML Vector Migration Pending Memory Helper Ownership Development 20260515

## Goal

Continue Phase 3 helper ownership cleanup by moving the in-memory branch of
pending candidate collection out of the shared memory/Qdrant entrypoint.

This slice targets exact in-memory pending candidate scanning, including
missing-vector skips, version filtering, item construction, page-limit behavior,
and complete count/distribution reporting for migration planning.

## Changes

- Added `src/core/vector_migration_pending_memory.py` with
  `collect_memory_migration_pending_candidates`.
- Updated `src/core/vector_migration_pending_candidates.py` so
  `collect_vector_migration_pending_candidates` delegates its memory branch to
  the new helper.
- Kept the public shared entrypoint unchanged:
  `collect_vector_migration_pending_candidates`.
- Kept the API facade wrapper unchanged:
  `src.api.v1.vectors._collect_vector_migration_pending_candidates`.
- Added `tests/unit/test_vector_migration_pending_memory.py` for:
  - exact pending counts independent from the page limit.
  - bounded `items` and returned `pending_ids`.
  - normalized filter application.
  - `unknown` feature-version handling for empty or missing metadata.
  - skipping metadata entries missing from `vector_store`.
- Extended `tests/unit/test_vector_migration_pending_candidates.py` to prove the
  shared entrypoint delegates memory requests with normalized filters and the
  expected arguments.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Compatibility Notes

- Runtime API behavior is unchanged.
- Existing code that patches or calls
  `src.api.v1.vectors._collect_vector_migration_pending_candidates` remains
  supported.
- `collect_vector_migration_pending_candidates` still owns
  `from_version_filter` normalization before entering either backend branch.
- The memory helper intentionally accepts `normalized_filter`; direct callers
  should pass `None` for no filter and use the shared entrypoint when accepting
  raw user input.
- `scan_limit` is echoed for response compatibility in memory results, but it
  does not bound the in-memory scan. Memory results remain exact, so
  `distribution_complete` is always `True` and `total_pending` is complete.
- The Qdrant branch remains delegated to
  `src/core/vector_migration_pending_qdrant.py` and was not changed.

## Claude Code Assistance

Claude Code was used as a read-only compatibility reviewer.

Invocation boundary:

- stdin-only targeted snippets.
- no tools enabled.
- no file writes.
- no secrets or environment dumps.
- limited budget.

Reviewer result: compatible. Claude Code confirmed behavior preservation,
filter normalization ownership, missing-vector skip semantics, exact memory
count semantics, `pending_ids[:limit]`, item capping, and API facade delegation.
It also recommended documenting that direct helper callers must pre-normalize
filters and that `scan_limit` is a memory-branch response-compatibility echo.

## Remaining Work

- Continue helper ownership cleanup only where a backend branch or helper has
  clear ownership and focused tests.
- Preserve `src.api.v1.vectors.*` facade wrappers while split routers still use
  them as monkeypatch surfaces.
- Keep migration pending/plan/run tests in the target validation set when
  changing pending-candidate behavior.
