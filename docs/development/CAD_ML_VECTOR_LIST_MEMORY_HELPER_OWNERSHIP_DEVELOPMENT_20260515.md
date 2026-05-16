# CAD ML Vector List Memory Helper Ownership Development 20260515

## Goal

Continue Phase 3 vector-router closeout by moving in-memory vector-list storage
logic out of `src/api/v1/vectors.py` while preserving the existing facade patch
surface.

This slice targets `_list_vectors_memory`, which converts in-memory vector and
metadata dictionaries into `VectorListResponse` payloads.

## Changes

- Added `src/core/vector_list_memory.py` with `list_vectors_memory`.
- Replaced the local memory-list implementation in `src/api/v1/vectors.py` with
  a thin `_list_vectors_memory` wrapper.
- The wrapper injects:
  - `VectorListItem`
  - `VectorListResponse`
  - `src.api.v1.vectors._matches_vector_label_filters`
- Added `tests/unit/test_vector_list_memory.py` for:
  - pagination and item construction.
  - filter argument propagation.
  - facade label-filter monkeypatch compatibility.
- Kept existing list route delegation coverage proving
  `vectors_module._list_vectors_memory` is passed into the shared list pipeline.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Compatibility Notes

- Runtime API behavior is unchanged.
- Existing callers and tests that patch
  `src.api.v1.vectors._list_vectors_memory` remain supported.
- Existing callers and tests that patch
  `src.api.v1.vectors._matches_vector_label_filters` still affect the memory
  list facade wrapper.
- The core helper accepts `item_cls`, `response_cls`, and
  `matches_label_filters_fn`, so it remains independent of FastAPI response
  models.

## Claude Code Assistance

Claude Code was used as a read-only compatibility reviewer.

Invocation boundary:

- stdin-only targeted snippets.
- no tools enabled.
- no file writes.
- no secrets or environment dumps.
- limited budget.

Reviewer result: compatible. Claude Code confirmed the wrapper reads the facade
filter helper at call time and list route delegation still passes
`vectors_module._list_vectors_memory` into the shared pipeline.

## Remaining Work

- Continue helper ownership cleanup in small slices.
- Keep route-level delegation tests active while `src.api.v1.vectors` remains a
  compatibility facade.
- The next natural candidate is the Redis vector-list storage helper, which
  needs Redis scan/hgetall edge-case tests before migration.
