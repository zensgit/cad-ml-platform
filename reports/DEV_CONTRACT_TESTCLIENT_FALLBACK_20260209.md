# DEV_CONTRACT_TESTCLIENT_FALLBACK_20260209

## Goal
Make the contract test tier runnable in environments where binding a local HTTP port
is not permitted (while keeping CI behavior unchanged), and add basic contract
coverage for the new knowledge endpoints.

## Context
- The repo's contract tier is normally executed against a live uvicorn server via
  `scripts/test_with_local_api.sh --suite contract` (CI behavior).
- Some restricted local environments cannot bind `127.0.0.1:8000`, which blocks
  end-to-end HTTP contract runs even though the FastAPI app can be imported and
  exercised in-process.

## Changes
### 1) Live-HTTP + TestClient fallback for manual contract tests
File: `tests/contract/test_api_contract.py`
- Added `_live_server_available()` probe to determine whether `API_BASE_URL` is
  reachable.
- Added `_get_test_client()` and `_request()` helper:
  - If the live server is reachable, use `requests` against `API_BASE_URL`.
  - Otherwise, use `fastapi.testclient.TestClient` against `src.main:app` in-process.
- Refactored existing manual contract assertions to use `_request()` so they can
  run either mode.

Notes:
- Schemathesis-based tests remain live-HTTP only; when the OpenAPI schema cannot
  be loaded from `API_BASE_URL`, those tests are skipped as before.

### 2) Knowledge endpoint contract coverage
File: `tests/contract/test_api_contract.py`
Added `TestKnowledgeApiContracts` covering response shape for:
- `GET /api/v1/tolerance/it`
- `GET /api/v1/tolerance/fit`
- `GET /api/v1/standards/status`
- `GET /api/v1/standards/thread`

## Validation
Commands run:
```bash
pytest tests/contract/test_api_contract.py -q
pytest tests/unit/test_provider_registry_bootstrap.py -q
pytest tests/unit/test_provider_framework_knowledge_bridge.py -q
```

Results:
- Contract suite: `13 passed, 4 skipped` (skips correspond to schemathesis/schema
  availability; manual contract assertions execute via TestClient fallback when
  the live server is not reachable).
- Provider framework unit suites: passed.

## Operational Notes
- CI should continue to exercise the contract tier against a live uvicorn server
  (when `API_BASE_URL` is reachable).
- Local developers in restricted environments can still validate the contract
  assertions using the in-process fallback without changing CI scripts.

