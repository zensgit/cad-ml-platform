#!/usr/bin/env markdown
# Dev E2E Smoke Report (Week 5) - 2025-12-22

## Scope
- API E2E smoke: analyze, vectors register/search/list/stats, knowledge status.
- Dedup 2D search smoke (PNG fixture).

## Test
- Command:
  `pytest tests/integration/test_e2e_api_smoke.py -q`
- Result: `2 passed in 21.94s`

## Notes
- Test auto-skips if API is unavailable or fixtures are missing.
- Configure `API_BASE_URL`, `API_KEY`, `E2E_DXF_PATH`, `E2E_PNG_PATH` to enable.
