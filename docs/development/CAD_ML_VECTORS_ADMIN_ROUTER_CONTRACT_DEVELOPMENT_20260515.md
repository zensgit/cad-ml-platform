# CAD ML Vectors Admin Router Contract Development

Date: 2026-05-15

## Goal

Close the P0 `phase3-vectors-admin-router` slice by pinning the backend reload
route contract after extraction into `src/api/v1/vectors_admin_router.py`.

The runtime route had already been split; this slice adds stronger regression
guards so future cleanup cannot silently re-register the route in `vectors.py`,
change the OpenAPI operationId, change the response model, or drop the admin
token requirement.

## Changes

- Updated `tests/unit/test_vectors_admin_router.py`.
  - Added a route-count guard for `POST /api/v1/vectors/backend/reload`.
  - Asserted the route exposes only `POST`.
  - Asserted there is no trailing-slash duplicate route.
  - Pinned the OpenAPI `operationId`:
    `reload_vector_backend_api_v1_vectors_backend_reload_post`.
  - Pinned global uniqueness for that operationId.
  - Pinned the OpenAPI summary, tag, query/header parameters, lack of
    request body, and `VectorBackendReloadResponse` response schema ref.
  - Added an explicit unauthenticated/admin-token rejection check.
  - Expanded the facade-leak guard to include both `/api/v1/vectors` and
    `/api/v1/vectors/`.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.
  - Marked the admin-router contract guard as complete.

## Claude Code Review

Claude Code is available locally as `/Users/chouhua/.local/bin/claude` and was
used as a read-only reviewer for the vectors facade, admin router, and contract
test snippets. Tools were disabled and no secrets or environment values were
sent.

Claude Code confirmed the original ownership and operationId checks were useful,
then identified high-value gaps around duplicate registration, trailing slash
duplicates, response schema drift, operationId uniqueness, and admin auth
contract coverage. This slice added those assertions.

## Release Impact

No runtime behavior changed. The added tests protect the admin reload API
contract while preserving the existing path, response model, auth behavior, and
OpenAPI snapshot.

## Remaining Work

- Keep `vectors.py` helper ownership cleanup separate until compatibility tests
  are stable.
- Continue Phase 3 router closeout with list/similarity helpers only in small
  PR-sized slices.
- Do not refresh the OpenAPI snapshot unless a deliberate API change is made.
