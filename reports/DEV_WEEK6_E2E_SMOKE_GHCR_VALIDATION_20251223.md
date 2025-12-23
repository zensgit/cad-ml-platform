# DEV_WEEK6_E2E_SMOKE_GHCR_VALIDATION_20251223

## Objective
Run E2E smoke tests using the GHCR-pinned dedupcad-vision image and verify API integration locally.

## Environment
- API: uvicorn `src.main:app` on `http://localhost:8000`
- dedupcad-vision: `ghcr.io/zensgit/dedupcad-vision@sha256:41cd67e8f7aeeb2a96b5fa3c49797af79ee4dda4df9885640a1385826cbbe5ce`
- DEDUPCAD_VISION_REQUIRED=1

## Steps
1. Start dedupcad-vision container on port 58001.
2. Start API with `DEDUPCAD_VISION_URL=http://localhost:58001` and `REDIS_ENABLED=false`.
3. Run:
   - `make e2e-smoke PYTHON=.venv/bin/python`

## Results
- `tests/integration/test_e2e_api_smoke.py::test_e2e_core_api_smoke` PASSED
- `tests/integration/test_e2e_api_smoke.py::test_e2e_dedup_search_smoke` PASSED
- `tests/integration/test_dedupcad_vision_contract.py::test_vision_health_contract` PASSED
- `tests/integration/test_dedupcad_vision_contract.py::test_vision_search_contract` PASSED

## Notes
- Initial health check hit one transient "connection reset by peer" before the container was ready; subsequent checks passed.
