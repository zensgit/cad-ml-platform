# CAD ML Vector Migration Upgrade Helper Ownership Development 20260515

## Goal

Continue Phase 3 vector-router closeout by moving vector migration upgrade
preparation logic out of `src/api/v1/vectors.py` while preserving the existing
facade patch surface.

This slice targets `_prepare_vector_for_upgrade`, which normalizes vectors for
feature-version migration and preserves L3 tails.

## Changes

- Added `src/core/vector_migration_upgrade.py` with
  `prepare_vector_for_upgrade`.
- Re-exported the helper through the existing facade name:
  `src.api.v1.vectors._prepare_vector_for_upgrade`.
- Removed the local helper implementation from `src/api/v1/vectors.py`.
- Added `tests/unit/test_vector_migration_upgrade.py` for:
  - legacy layout reorder behavior.
  - explicit L3 tail preservation.
  - inferred L3 tail dimensions.
  - invalid L3 layout rejection.
  - facade identity compatibility.
- Strengthened split-router delegation tests so preview and migrate routes prove
  they pass the facade helper into their pipelines at request time.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Compatibility Notes

- Runtime API behavior is unchanged.
- Existing callers and tests that patch
  `src.api.v1.vectors._prepare_vector_for_upgrade` remain supported.
- `vectors_migration_read_router.preview_migration` and
  `vectors_write_router.migrate_vectors` still resolve the helper through
  `src.api.v1.vectors`.
- The new core helper can be tested without importing the full vector API
  facade.

## Claude Code Assistance

Claude Code was used as a read-only compatibility reviewer.

Invocation boundary:

- stdin-only targeted snippets.
- no tools enabled.
- no file writes.
- no secrets or environment dumps.
- limited budget.

Reviewer result: compatible. Claude Code confirmed the route handlers read
`vectors_module._prepare_vector_for_upgrade` at request time and pass that
callable into the pipelines, so facade-level monkeypatching still works.

## Remaining Work

- Continue helper ownership cleanup in small slices.
- Keep route-level delegation tests active while `src.api.v1.vectors` remains a
  compatibility facade.
- Remaining candidates include memory/Redis list helpers and migration read
  collection helpers, each requiring facade compatibility tests.
