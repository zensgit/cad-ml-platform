# CAD ML Vector Migration Qdrant Versions Helper Ownership Development 20260515

## Goal

Continue Phase 3 vector-router closeout by moving Qdrant feature-version
collection logic out of `src/api/v1/vectors.py` while preserving the existing
facade patch surface.

This slice targets `_collect_qdrant_feature_versions`, which scans Qdrant vector
metadata and returns feature-version counts plus scan coverage.

## Changes

- Added `src/core/vector_migration_qdrant_versions.py` with
  `collect_qdrant_feature_versions`.
- Replaced the local collector implementation in `src/api/v1/vectors.py` with a
  thin `_collect_qdrant_feature_versions` wrapper.
- The wrapper injects
  `src.api.v1.vectors._resolve_vector_migration_scan_limit` at call time, so
  facade scan-limit monkeypatch behavior remains intact.
- Added `tests/unit/test_vector_migration_qdrant_versions.py` for:
  - version counting.
  - unknown-version fallback.
  - default scan-limit resolver injection.
  - scan-limit clamping.
  - facade resolver patch compatibility.
  - explicit facade `scan_limit` pass-through without resolver calls.
- Strengthened reporting and trends route delegation tests to prove split routes
  still pass the facade collector into shared pipelines.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Compatibility Notes

- Runtime API behavior is unchanged.
- Existing callers and tests that patch
  `src.api.v1.vectors._collect_qdrant_feature_versions` remain supported.
- Existing callers and tests that patch
  `src.api.v1.vectors._resolve_vector_migration_scan_limit` still affect the
  facade collector when `scan_limit` is omitted.
- Direct core callers can pass `resolve_scan_limit_fn` explicitly when they need
  custom scan-limit resolution.

## Claude Code Assistance

Claude Code was used as a read-only compatibility reviewer.

Invocation boundary:

- stdin-only targeted snippets.
- no tools enabled.
- no file writes.
- no secrets or environment dumps.
- limited budget.

Reviewer result: compatible. Claude Code confirmed the facade wrapper resolves
the scan-limit helper at call time and split routes still pass
`vectors_module._collect_qdrant_feature_versions` into downstream pipelines.
It also flagged that explicit `scan_limit` pass-through deserved coverage; this
was addressed in `tests/unit/test_vector_migration_qdrant_versions.py`.

## Remaining Work

- Continue helper ownership cleanup in small slices.
- Keep route-level delegation tests active while `src.api.v1.vectors` remains a
  compatibility facade.
- Remaining candidates include Qdrant preview sample collection and memory/Redis
  vector-list storage helpers.
