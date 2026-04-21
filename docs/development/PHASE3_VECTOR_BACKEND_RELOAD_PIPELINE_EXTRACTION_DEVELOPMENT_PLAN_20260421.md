# Phase 3 Vector Backend Reload Pipeline Extraction Development Plan

## Goal

Extract the `/api/v1/vectors/backend/reload` execution body into a shared core helper while preserving the existing admin-token dependency and response contract.

## Scope

- Add a shared backend reload helper under `src/core/`
- Keep `vectors.py` responsible for route signature and dependency injection
- Preserve existing behavior for:
  - invalid backend validation
  - environment update
  - reload failure/error handling
  - reload metrics

## Files

- `src/core/vector_backend_reload_pipeline.py`
- `src/api/v1/vectors.py`
- `tests/unit/test_vector_backend_reload_pipeline.py`
- `tests/unit/test_vectors_backend_reload_delegation.py`

## Risk Controls

- Do not change `_vector_reload_admin_token`
- Do not change response model or route path
- Validate with existing vectors reload endpoint tests in addition to new helper tests
