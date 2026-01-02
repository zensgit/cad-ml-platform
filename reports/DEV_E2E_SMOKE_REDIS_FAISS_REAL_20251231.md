# Redis/FAISS E2E Smoke (Real FAISS) (2025-12-31)

## Scope

- Validate E2E smoke against a local API using Redis + FAISS backend with a real FAISS install.

## Environment

- Redis: local `redis-server`
- FAISS: `faiss-cpu 1.13.2`
- API: `VECTOR_STORE_BACKEND=faiss`, `REDIS_URL=redis://localhost:6379/0`
- dedupcad-vision: `scripts/dedupcad_vision_stub.py`
- API URL: `http://localhost:8002`
- Vision URL: `http://localhost:58001`

## Command

- `API_BASE_URL=http://localhost:8002 DEDUPCAD_VISION_URL=http://localhost:58001 make e2e-smoke`

## Results

- Passed: 4
- Failed: 0
- Duration: 3.91s

## Notes

- API logs confirm FAISS loaded successfully.
