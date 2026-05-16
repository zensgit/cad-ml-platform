# CAD ML Vector List Qdrant Helper Ownership Development 20260515

## Goal

Continue Phase 3 vector-list cleanup by moving the Qdrant-backed list branch out
of `src/core/vector_list_pipeline.py` into a focused core helper.

This slice targets the `source=qdrant` path, which builds native filter
conditions, calls `qdrant_store.list_vectors(..., with_vectors=True)`, converts
Qdrant results into vector-list response items, and preserves fallback behavior
when no Qdrant store is available.

## Changes

- Added `src/core/vector_list_qdrant.py` with async `list_vectors_qdrant`.
- Replaced the inline Qdrant branch body in `src/core/vector_list_pipeline.py`
  with a helper call.
- Preserved the existing pipeline responsibilities:
  - source validation.
  - `VECTOR_LIST_LIMIT` clamping.
  - source/backend resolution.
  - Qdrant store discovery.
  - fallback to Redis or memory when Qdrant is unavailable.
- Kept `build_filter_conditions_fn` injected by the caller and passed through to
  the Qdrant helper.
- Kept `extract_vector_label_contract` resolved inside the Qdrant branch and
  passed explicitly to the helper, preserving call-time patch behavior.
- Added `tests/unit/test_vector_list_qdrant.py` for:
  - filter builder argument propagation.
  - Qdrant `with_vectors=True` list call behavior.
  - response item construction.
  - missing metadata and missing vector fallback behavior.
  - raw metadata delivery to the extractor.
  - extractor monkeypatch compatibility through `run_vector_list_pipeline`.
  - fallback to memory when `source=qdrant` has no Qdrant store.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Compatibility Notes

- Runtime API behavior is unchanged.
- Existing route-level monkeypatches of
  `src.api.v1.vectors._build_vector_filter_conditions` still flow into
  `run_vector_list_pipeline` and then into the Qdrant helper.
- `src.core.similarity.extract_vector_label_contract` remains a call-time
  dependency in the pipeline branch, so tests that patch that function remain
  effective.
- `list_vectors_qdrant` intentionally requires `extract_label_contract_fn`
  explicitly to avoid future module-import-time patch surprises.

## Claude Code Assistance

Claude Code was used as a read-only compatibility reviewer.

Invocation boundary:

- stdin-only targeted snippets.
- no tools enabled.
- no file writes.
- no secrets or environment dumps.
- limited budget.

Reviewer result: compatible. The first pass recommended adding explicit fallback
coverage and making the extractor dependency explicit. Both changes were applied.
A final pass found no remaining compatibility issues and recommended a direct
call-site grep, which confirmed only the pipeline and the new tests call the
helper.

## Remaining Work

- Continue helper ownership cleanup only where the helper has clear ownership
  and focused tests.
- Keep OpenAPI and route-delegation checks in the target validation set.
- Avoid broader vector router rewrites until the remaining thin facade helpers
  have either dedicated compatibility tests or are intentionally left as facade
  shims.
