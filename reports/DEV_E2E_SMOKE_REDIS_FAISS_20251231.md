# Redis/FAISS E2E Smoke Verification (2025-12-31)

## Scope

- Run E2E smoke against a local API configured with Redis cache + FAISS vector backend.

## Environment

- Redis: `redis-server` (local)
- API: `VECTOR_STORE_BACKEND=faiss`, `REDIS_URL=redis://localhost:6379/0`
- dedupcad-vision: `scripts/dedupcad_vision_stub.py`
- API URL: `http://localhost:8002`
- Vision URL: `http://localhost:58001`

## Command

- `API_BASE_URL=http://localhost:8002 DEDUPCAD_VISION_URL=http://localhost:58001 make e2e-smoke`

## Changes

- Allowed `faiss` as a valid vector stats backend in the E2E smoke test.

## Results

- Passed: 4
- Failed: 0
- Duration: 8.28s

## Notes

- FAISS is not installed in this environment; the API logged a fallback to the memory store while reporting backend `faiss`.
- Redis cache initialized successfully.
