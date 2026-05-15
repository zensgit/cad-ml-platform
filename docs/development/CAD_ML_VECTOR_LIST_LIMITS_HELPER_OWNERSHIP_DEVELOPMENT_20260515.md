# CAD ML Vector List Limits Helper Ownership Development 20260515

## Goal

Continue Phase 3 vector-list cleanup by moving vector-list limit resolution out
of `src/core/vector_list_pipeline.py` into a focused, directly testable helper.

This slice targets the `VECTOR_LIST_LIMIT` and `VECTOR_LIST_SCAN_LIMIT` parsing
that controls response limit clamping and Redis scan bounds for list requests.

## Changes

- Added `src/core/vector_list_limits.py` with `resolve_vector_list_limits`.
- Replaced inline env parsing in `src/core/vector_list_pipeline.py` with the new
  helper.
- Preserved existing behavior:
  - `VECTOR_LIST_LIMIT` defaults to `200`.
  - requested list `limit` is clamped with `min(requested_limit, max_limit)`.
  - `VECTOR_LIST_SCAN_LIMIT` defaults to `5000`.
  - scan limit is still passed to the Redis list branch.
  - invalid env values still fail through `int(...)`.
- Added `tests/unit/test_vector_list_limits.py` for default values, env override,
  invalid list limit, and invalid scan limit.
- Extended `tests/unit/test_vector_list_pipeline.py` to verify the pipeline
  applies clamped `limit` and passes `scan_limit` into the Redis branch.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Compatibility Notes

- Runtime API behavior is unchanged.
- The helper keeps the previous eager env parsing timing in
  `run_vector_list_pipeline`: both list limit and scan limit are resolved before
  source dispatch.
- Invalid env handling intentionally remains strict because the previous inline
  implementation used direct `int(os.getenv(...))` conversion.
- No route-level monkeypatch surface was removed.

## Claude Code Assistance

Claude Code was used as a read-only compatibility reviewer.

Invocation boundary:

- stdin-only targeted snippets.
- no tools enabled.
- no file writes.
- no secrets or environment dumps.
- limited budget.

Reviewer result: compatible. Claude Code confirmed clamping and scan-limit
pass-through matched the prior inline logic. It flagged that the default-value
test should exercise missing env keys directly and that invalid scan-limit
failure semantics should be pinned. Both recommendations were applied.

## Remaining Work

- Continue helper ownership cleanup only where the behavior has clear ownership
  and focused tests.
- Keep route-delegation and OpenAPI checks in validation when touching list or
  vector router contracts.
- Leave thin facade wrappers in place unless there is a concrete compatibility
  test proving they can be removed.
