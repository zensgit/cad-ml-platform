# E2E Smoke Vector Search Filter Fix (2025-12-31)

## Scope

- Stabilize the core E2E smoke test when vector search results contain duplicate vectors.

## Changes

- Use unique `material` and `complexity` values derived from the generated vector id.
- Pass those filters into `/api/v1/vectors/search` to ensure the registered vector is discoverable.

## Rationale

- Repeated runs can leave identical vectors in the store; a small `k` search may return older ids and omit the newly registered one.
