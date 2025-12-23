# Week6 Step1 - E2E Regression Smoke (2025-12-22)

## Scope
- New pytest-based E2E smoke suite for core API + dedup 2D.
- Targets a running cad-ml-platform at `API_BASE_URL` (default `http://localhost:8000`).

## Tests
- `pytest tests/integration/test_e2e_api_smoke.py -q`

## Results
- `2 passed in 21.94s`

## Notes
- The suite validates analyze -> combined features, vector register/search/list/stats, knowledge status, and dedup 2D search.
