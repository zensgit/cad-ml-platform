# CAD ML Vector Filtering Helper Ownership Development

Date: 2026-05-15

## Goal

Start the Phase 3 helper ownership cleanup after stabilizing split-router
contracts. This slice moves the pure vector filtering helper implementations out
of `src/api/v1/vectors.py` into core code while preserving the
`src.api.v1.vectors.*` facade attributes used by existing monkeypatch tests and
split routers.

## Changes

- Added `src/core/vector_filtering.py`.
  - `build_vector_filter_conditions`
  - `build_vector_search_filter_conditions`
  - `vector_item_payload`
  - `matches_vector_label_filters`
  - `matches_vector_search_filters`
- Updated `src/api/v1/vectors.py`.
  - Re-imports the core helpers under the existing underscored facade names:
    - `_build_vector_filter_conditions`
    - `_build_vector_search_filter_conditions`
    - `_vector_item_payload`
    - `_matches_vector_label_filters`
    - `_matches_vector_search_filters`
  - Keeps split-router late binding through `vectors_module.<attr>` intact.
  - Reduced `vectors.py` to 468 lines in this local checkout.
- Added `tests/unit/test_vector_filtering.py`.
  - Covers filter condition construction, including false boolean filters.
  - Covers payload-derived search filters.
  - Covers vector payload shape.
  - Covers metadata and label-contract matching.
  - Covers facade helper export identity.
- Updated existing delegation tests.
  - List delegation still verifies late-bound facade helper wiring.
  - Batch-similarity delegation still verifies late-bound facade helper wiring.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.
  - Marked the vector filtering helper move as complete.
  - Kept remaining helper ownership cleanup as a separate small-slice task.

## Claude Code Review

Claude Code is available locally as `/Users/chouhua/.local/bin/claude` and was
used as a read-only reviewer for the core helper, facade imports, affected list
code, and compatibility tests. Tools were disabled and no secrets or environment
values were sent.

Claude Code confirmed the facade monkeypatch contract is preserved when split
routers continue to read helpers through `vectors_module.<attr>` at request
time. It also confirmed the moved helper behavior matches the previous inline
implementation and highlighted the same late-binding contract now covered by the
delegation tests.

## Release Impact

No API behavior changed. This is an ownership cleanup that moves pure helper
logic into core while keeping existing route behavior, operationIds, schema
contracts, and monkeypatch paths intact.

## Remaining Work

- Continue helper ownership cleanup in small slices only after compatibility
  tests stay green.
- Avoid moving Redis/list/migration helper logic until its monkeypatch and
  OpenAPI contracts are pinned.
- Do not refresh the OpenAPI snapshot unless a deliberate API change is made.
