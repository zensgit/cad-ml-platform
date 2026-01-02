# CI DedupCAD Vision Integration Report

- Date: 2025-12-27
- Scope: CI e2e-smoke job (dedupcad-vision image pull + traceability)

## Changes
- Added an explicit image pull step for dedupcad-vision in CI.
- Recorded the image reference + ID in the job summary for traceability.

## Local Verification
- dedupcad-vision started via `python start_server.py --port 58001`
- cad-ml-platform API available at `http://localhost:8000`

### Commands
- pytest tests/integration/test_dedupcad_vision_contract.py -q
- pytest tests/integration/test_e2e_api_smoke.py::test_e2e_dedup_search_smoke -q

## Result
- PASS

## Summary
- Contract tests: 2 passed in 2.92s
- E2E dedup search smoke: 1 passed in 0.64s
