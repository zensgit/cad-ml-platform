# CAD ML Vector List Redis Helper Ownership Development 20260515

## Goal

Continue Phase 3 vector-router closeout by moving Redis-backed vector-list
storage logic out of `src/api/v1/vectors.py` while preserving the existing API
facade patch surface.

This slice targets `_list_vectors_redis`, which scans Redis `vector:*` hashes,
decodes vector metadata, applies list filters, and builds `VectorListResponse`
payloads.

## Changes

- Added `src/core/vector_list_redis.py` with async `list_vectors_redis`.
- Replaced the local Redis list implementation in `src/api/v1/vectors.py` with
  a thin `_list_vectors_redis` wrapper.
- The wrapper injects:
  - `VectorListItem`
  - `VectorListResponse`
  - `src.api.v1.vectors._matches_vector_label_filters`
  - call-time `src.core.similarity.extract_vector_label_contract`
  - call-time `json.loads`
- Added `tests/unit/test_vector_list_redis.py` for:
  - pagination and item construction.
  - Redis scan filter arguments.
  - `scan_limit` hgetall cutoff behavior.
  - malformed JSON and non-dict metadata fallbacks.
  - multi-batch scan cursor continuation.
  - facade label-filter monkeypatch compatibility.
  - facade extractor and `json.loads` patch compatibility.
- Kept existing list route delegation coverage proving
  `vectors_module._list_vectors_redis` is passed into the shared list pipeline.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Compatibility Notes

- Runtime API behavior is unchanged.
- Existing callers and tests that patch
  `src.api.v1.vectors._list_vectors_redis` remain supported.
- Existing callers and tests that patch
  `src.api.v1.vectors._matches_vector_label_filters` still affect the Redis
  list facade wrapper.
- The wrapper resolves `extract_vector_label_contract` at call time, matching
  the previous local-import behavior.
- The wrapper forwards `json.loads`, preserving the prior facade-level JSON
  patch surface.
- The core helper remains independent of FastAPI response models by accepting
  `item_cls`, `response_cls`, and matcher/extractor/parser callables.

## Claude Code Assistance

Claude Code was used as a read-only compatibility reviewer.

Invocation boundary:

- stdin-only targeted snippets.
- no tools enabled.
- no file writes.
- no secrets or environment dumps.
- limited budget.

Reviewer result: compatible with one actionable recommendation set. Claude Code
confirmed the facade matcher is late-bound, then flagged that extractor/JSON
patch surfaces and malformed/multi-batch Redis branches should be pinned. This
slice incorporated those recommendations with call-time dependency injection and
additional tests.

## Remaining Work

- Continue helper ownership cleanup in small slices.
- Keep route-level delegation tests active while `src.api.v1.vectors` remains a
  compatibility facade.
- The next natural candidate is to keep reducing the remaining route facade
  surface only where a helper has clear ownership and focused tests.
