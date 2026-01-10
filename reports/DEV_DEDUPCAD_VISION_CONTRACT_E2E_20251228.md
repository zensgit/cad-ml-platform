# DedupCAD Vision Contract + E2E Report

- Date: 2025-12-28
- Scope: `tests/integration/test_dedupcad_vision_contract.py`, `tests/integration/test_e2e_api_smoke.py`

## Setup
- dedupcad-vision server: `python start_server.py --port 58001`
- Env:
  - `DEDUPCAD_VISION_URL=http://localhost:58001`
  - `DEDUPCAD_VISION_REQUIRED=1`

## Commands
- `DEDUPCAD_VISION_URL=http://localhost:58001 DEDUPCAD_VISION_REQUIRED=1 .venv/bin/python -m pytest tests/integration/test_dedupcad_vision_contract.py -q`
- `DEDUPCAD_VISION_URL=http://localhost:58001 DEDUPCAD_VISION_REQUIRED=1 .venv/bin/python -m pytest tests/integration/test_e2e_api_smoke.py -q`

## Results
- Contract tests: 2 passed
- E2E smoke: 2 passed
