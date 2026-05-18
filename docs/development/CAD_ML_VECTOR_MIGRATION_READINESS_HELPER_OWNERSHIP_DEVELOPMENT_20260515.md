# CAD ML Vector Migration Readiness Helper Ownership Development 20260515

## Goal

Continue Phase 3 vector-router closeout by moving vector migration readiness
calculation out of `src/api/v1/vectors.py` while preserving the existing facade
patch surface.

This slice targets `_build_vector_migration_readiness`, which converts a feature
version distribution into target-version readiness fields.

## Changes

- Added `src/core/vector_migration_readiness.py` with
  `build_vector_migration_readiness`.
- Replaced the local readiness implementation in `src/api/v1/vectors.py` with a
  thin `_build_vector_migration_readiness` wrapper.
- The wrapper injects
  `src.api.v1.vectors._resolve_vector_migration_target_version` at call time, so
  facade target-version monkeypatch behavior remains intact.
- Added `tests/unit/test_vector_migration_readiness.py` for:
  - complete distribution readiness.
  - partial distribution readiness.
  - empty total readiness.
  - facade target-version resolver patch compatibility.
- Strengthened reporting and trends route delegation tests to prove split routes
  still pass the facade readiness helper into shared pipelines.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Compatibility Notes

- Runtime API behavior is unchanged.
- Existing callers and tests that patch
  `src.api.v1.vectors._build_vector_migration_readiness` remain supported.
- Existing callers and tests that patch
  `src.api.v1.vectors._resolve_vector_migration_target_version` still affect the
  facade readiness wrapper.
- Direct core callers can use `resolve_target_version_fn` explicitly when they
  need custom target-version resolution.

## Claude Code Assistance

Claude Code was used as a read-only compatibility reviewer.

Invocation boundary:

- stdin-only targeted snippets.
- no tools enabled.
- no file writes.
- no secrets or environment dumps.
- limited budget.

Reviewer result: compatible. Claude Code confirmed the facade wrapper resolves
the target-version helper at call time and split routes still pass
`vectors_module._build_vector_migration_readiness` into downstream pipelines.

## Remaining Work

- Continue helper ownership cleanup in small slices.
- Keep route-level delegation tests active while `src.api.v1.vectors` remains a
  compatibility facade.
- Remaining higher-touch candidates include memory/Redis vector-list storage
  helpers and Qdrant migration collection helpers.
