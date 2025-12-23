# E2E Smoke Validation (Week 7)

## Summary
- Verified `dedupcad-vision` multi-arch image boots and the API integrates correctly in local e2e smoke tests.

## Test Setup
- dedupcad-vision container:
  - Image: `ghcr.io/zensgit/dedupcad-vision@sha256:9f7f567e3b0c1c882f9a363f1b1cb095d30d9e9b184e582d6b19ec7446a86251`
  - Port: `58001:8000`
  - Env: `S3_ENABLED=false`, `EVENT_BUS_ENABLED=false`, `ML_PLATFORM_ENABLED=false`, `GEOMETRIC_ENABLED=false`, `VISION_ENV=ci`
- cad-ml-platform API:
  - `.venv/bin/python -m uvicorn src.main:app --host 0.0.0.0 --port 8000`
  - Env: `REDIS_ENABLED=false`, `DEDUPCAD_VISION_URL=http://localhost:58001`, `DEDUPCAD_VISION_TIMEOUT_SECONDS=10`

## Tests
- `.venv/bin/python -m pytest tests/integration/test_e2e_api_smoke.py tests/integration/test_dedupcad_vision_contract.py -v -rs`
  - Result: PASS (4 passed)

## Notes
- Initial health probe occasionally returns `curl: (56) Recv failure: Connection reset by peer` before the service is ready; subsequent probes succeed.
