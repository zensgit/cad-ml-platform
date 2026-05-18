# CAD ML Vector Migration Qdrant Preview Helper Ownership Development 20260515

## Goal

Continue Phase 3 vector-router closeout by moving Qdrant migration preview
sample collection out of `src/api/v1/vectors.py` while preserving the existing
facade patch surface.

This slice targets `_collect_qdrant_preview_samples`, which gathers preview
vectors and a feature-version distribution for `/api/v1/vectors/migrate/preview`.

## Changes

- Added `src/core/vector_migration_qdrant_preview.py` with
  `collect_qdrant_preview_samples`.
- Re-exported the helper through the existing facade name:
  `src.api.v1.vectors._collect_qdrant_preview_samples`.
- Removed the local preview collector implementation from `src/api/v1/vectors.py`.
- Added `tests/unit/test_vector_migration_qdrant_preview.py` for:
  - preview sample collection.
  - `id`, vector, and metadata normalization.
  - feature-version distribution across sampled and non-sampled vectors.
  - initial limit clamping.
  - 200-item distribution scan batching.
  - facade identity compatibility.
- Strengthened the preview route delegation test to prove the route still passes
  the facade collector into the shared preview pipeline at request time.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Compatibility Notes

- Runtime API behavior is unchanged.
- Existing callers and tests that patch
  `src.api.v1.vectors._collect_qdrant_preview_samples` remain supported.
- The preview route continues to resolve the helper through
  `src.api.v1.vectors`, not by importing the core helper directly.
- Patching the core helper path does not affect the route; patch the facade path
  while `src.api.v1.vectors` remains the compatibility surface.

## Claude Code Assistance

Claude Code was used as a read-only compatibility reviewer.

Invocation boundary:

- stdin-only targeted snippets.
- no tools enabled.
- no file writes.
- no secrets or environment dumps.
- limited budget.

Reviewer result: compatible. Claude Code confirmed the preview route reads
`vectors_module._collect_qdrant_preview_samples` at request time, so
facade-level monkeypatching is preserved.

## Remaining Work

- Continue helper ownership cleanup in small slices.
- Keep route-level delegation tests active while `src.api.v1.vectors` remains a
  compatibility facade.
- Remaining candidates include memory and Redis vector-list storage helpers.
