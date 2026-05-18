# CAD ML Vectors List Similarity Router Contract Development

Date: 2026-05-15

## Goal

Continue Phase 3 router closeout without moving shared helpers yet. The list and
batch-similarity vector routes were already split into dedicated routers; this
slice pins their route and facade-compatibility contracts so later helper
ownership cleanup has a stable test baseline.

## Changes

- Updated `tests/unit/test_vectors_list_router.py`.
  - Added route single-registration and HTTP method checks for
    `GET /api/v1/vectors/`.
  - Asserted the no-slash variant is not a separately registered route.
  - Pinned OpenAPI operationId, summary, tag, response codes, query/header
    parameters, lack of request body, and `VectorListResponse` schema ref.
- Updated `tests/unit/test_vectors_similarity_router.py`.
  - Added route single-registration, HTTP method, and no trailing-slash duplicate
    checks for `POST /api/v1/vectors/similarity/batch`.
  - Pinned OpenAPI operationId, summary, tag, response codes, `X-API-Key`
    parameter, required request body, `BatchSimilarityRequest`, and
    `BatchSimilarityResponse`.
- Updated delegation tests.
  - `test_vectors_list_delegation.py` now verifies query value pass-through for
    pagination and all filter fields.
  - It also verifies late-bound facade helper wiring for list-source resolution,
    Qdrant store lookup, filter building, Redis/memory list helpers, and Redis
    client lookup.
  - `test_vectors_batch_similarity_delegation.py` now verifies `top_k`,
    response item fields, model class identities, and late-bound facade helper
    wiring for Qdrant lookup and filter building.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.
  - Marked list/batch-similarity split-router contract guards as complete.

## Claude Code Review

Claude Code is available locally as `/Users/chouhua/.local/bin/claude` and was
used as a read-only reviewer for the list/similarity router and test snippets.
Tools were disabled and no secrets or environment values were sent.

Claude Code identified high-value gaps around facade ABI protection,
late-binding monkeypatch semantics, pass-through query values, response-code
contracts, request-body requiredness, and response field faithfulness. This
slice added tests for those points while preserving runtime behavior.

## Release Impact

No runtime behavior changed. The added tests reduce risk before moving any
remaining helper ownership out of `vectors.py`, preserving the existing
`src.api.v1.vectors.*` monkeypatch surface and OpenAPI contract.

## Remaining Work

- Keep shared helper ownership cleanup separate and small.
- Avoid refreshing the OpenAPI snapshot unless an API change is deliberate.
- Continue Phase 3 router closeout only after these compatibility tests stay
  green in the broader suite.
