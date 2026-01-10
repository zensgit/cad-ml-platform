# DedupCAD Vision Integration Report

- Date: 2025-12-27
- Scope: cad-ml-platform ↔ dedupcad-vision (contract + e2e)

## Services
- dedupcad-vision: `python start_server.py --port 58001` (health 200)
- cad-ml-platform API: `http://localhost:8000` (health 200)

## Commands
- pytest tests/integration/test_dedupcad_vision_contract.py -q
- pytest tests/integration/test_e2e_api_smoke.py::test_e2e_dedup_search_smoke -q

## Result
- PASS

## Summary
- Contract tests: 2 passed in 4.22s
- E2E dedup search smoke: 1 passed in 0.57s
