# DEV Vector Backend Parity Smoke (2025-12-22)

## Scope
- Validate `/api/v1/vectors` and `/api/v1/vectors_stats/stats` against Redis backend
- Confirm list/stats totals are consistent after registration
- Document env controls for scan limits

## Environment
- Redis: `VECTOR_STORE_BACKEND=redis`, `REDIS_URL=redis://localhost:16379/0`
- Memory: `VECTOR_STORE_BACKEND=memory`
- Redis container: `cad-ml-redis` (host port 16379)

## Procedure
1. Redis path: register two vectors, query `/api/v1/vectors?source=redis` and `/api/v1/vectors_stats/stats`.\n2. Memory path: register two vectors, query `/api/v1/vectors?source=memory` and `/api/v1/vectors_stats/stats`.\n3. Confirm totals match and return 200.\n4. Clean up Redis keys and in-memory store entries.

## Result
- Redis list total: 2; stats total: 2 (200/200)
- Memory list total: 2; stats total: 2 (200/200)

## Notes
- Cleanup removed the temporary `vector:smoke-*` keys.
- Scan limit envs documented in `README.md` and `docs/ARCHITECTURE_INDEX.md`.
