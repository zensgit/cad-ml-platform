# CAD ML Vector Migration Config Helper Ownership Development 20260515

## Goal

Continue Phase 3 vector-router closeout by moving pure vector migration
configuration helpers out of `src/api/v1/vectors.py` while preserving the
existing facade monkeypatch surface.

This slice covers:

- `_resolve_vector_migration_scan_limit`
- `_resolve_vector_migration_target_version`
- `_coerce_int`

## Changes

- Added `src/core/vector_migration_config.py` with:
  - `resolve_vector_migration_scan_limit`
  - `resolve_vector_migration_target_version`
  - `coerce_optional_int`
  - `ALLOWED_VECTOR_MIGRATION_TARGET_VERSIONS`
- Re-exported those helpers through the existing `src.api.v1.vectors` facade
  names:
  - `_resolve_vector_migration_scan_limit`
  - `_resolve_vector_migration_target_version`
  - `_coerce_int`
- Removed the local helper implementations and unused `os` import from
  `src/api/v1/vectors.py`.
- Added `tests/unit/test_vector_migration_config.py` for core behavior and
  facade identity.
- Added `tests/unit/test_vectors_migration_config_delegation.py` to pin that
  split migration routes still honor facade-level scan-limit and target-version
  patches.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Compatibility Notes

- Runtime API behavior is unchanged.
- Env vars are still read at call time:
  - `VECTOR_MIGRATION_SCAN_LIMIT`
  - `VECTOR_MIGRATION_TARGET_VERSION`
- Existing tests and callers that patch
  `src.api.v1.vectors._resolve_vector_migration_scan_limit` or
  `src.api.v1.vectors._resolve_vector_migration_target_version` remain
  supported.
- The split routers continue to resolve helpers through
  `src.api.v1.vectors`, not directly from core.

## Claude Code Assistance

Claude Code was used as a read-only compatibility reviewer.

Invocation boundary:

- stdin-only targeted snippets.
- no tools enabled.
- no file writes.
- no secrets or environment dumps.
- limited budget.

Reviewer result: the alias-based move is compatible for in-module callers and
env behavior. Claude Code flagged one useful risk: split routers must continue
using the `src.api.v1.vectors` facade if route-level monkeypatch compatibility
is required. This was verified with code search and pinned with
`tests/unit/test_vectors_migration_config_delegation.py`.

## Remaining Work

- Continue helper ownership cleanup in small slices.
- Keep facade exports until the router compatibility tests are intentionally
  retired.
- Next low-risk candidates are storage list helpers or migration upgrade helper
  ownership, each with route-level delegation tests before behavior changes.
