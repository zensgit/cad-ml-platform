# CAD ML Vector List Source Helper Ownership Development 20260515

## Goal

Continue the Phase 3 vector-router closeout by moving another pure helper out of
`src/api/v1/vectors.py` while preserving the established facade patch surface.

This slice targets `_resolve_list_source`, which decides the effective vector
list source for `source=auto`.

## Changes

- Added `src/core/vector_list_sources.py` with
  `resolve_vector_list_source(source, backend)`.
- Re-exported the helper through the existing API facade name:
  `src.api.v1.vectors._resolve_list_source`.
- Removed the local `_resolve_list_source` implementation from
  `src/api/v1/vectors.py`.
- Added `tests/unit/test_vector_list_sources.py` for:
  - `auto` source resolution for Redis, Qdrant, memory, and unknown backends.
  - explicit source preservation.
  - facade identity compatibility.
- Kept the existing list-route delegation monkeypatch path intact:
  `src.api.v1.vectors._resolve_list_source`.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Compatibility Notes

- Runtime API behavior is unchanged.
- Existing tests that monkeypatch `src.api.v1.vectors._resolve_list_source`
  remain supported.
- The core helper is now independently testable without importing the full
  FastAPI router module.
- Future route changes should continue to resolve `_resolve_list_source` at
  request time instead of capturing it in a default argument.

## Claude Code Assistance

Claude Code was used as a read-only compatibility reviewer for this slice.

Invocation boundary:

- stdin-only code snippets.
- no tools enabled.
- no file writes.
- no secrets or environment dumps.
- limited budget.

Reviewer result: the alias-based move is compatibility-safe for the existing
facade identity and router monkeypatch contracts. The only caution is to keep
the route delegation late-bound.

## Remaining Work

- Continue shared helper ownership cleanup only in small, independently
  verified slices.
- Preserve facade exports until all known monkeypatch and route-contract tests
  have been migrated or intentionally retired.
