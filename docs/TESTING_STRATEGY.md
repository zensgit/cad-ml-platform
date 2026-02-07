# Testing Strategy

## Objective

This project uses tiered testing so failures are isolated quickly and CI remains actionable.

## Test Tiers

1. `unit`
- Scope: `tests/unit`
- Purpose: pure logic and module behavior without API bootstrap.
- Command: `make test-unit`

2. `contract-local`
- Scope: `tests/contract`
- Purpose: API response shape and schema-level behavior against a local running API.
- Command: `make test-contract-local`

3. `e2e-local`
- Scope: `tests/e2e`
- Purpose: request flow and endpoint behavior through Playwright API/browser clients.
- Command: `make test-e2e-local`

4. `all-local`
- Scope: `tests`
- Purpose: full-suite verification in one run.
- Command: `make test-all-local`

## Local API Orchestration

Use `scripts/test_with_local_api.sh` for local API-backed tiers:

```bash
bash scripts/test_with_local_api.sh --suite contract
bash scripts/test_with_local_api.sh --suite e2e
bash scripts/test_with_local_api.sh --suite all
```

Behavior:
- Reuses an already healthy API at `API_BASE_URL` if available.
- Otherwise starts `uvicorn src.main:app`, waits for `/health`, runs tests, and shuts down automatically.
- Exports `API_BASE_URL` and `API_KEY` for test processes.

## Endpoint Availability Policy

For `tests/e2e/test_api_e2e.py`, some `/api/v2/*` endpoints are optional depending on deployment profile.
Current policy:

1. Required baseline endpoints must return success (for example `/health`, `/api/v1/health`).
2. Optional `/api/v2/*` paths may return `404` and are treated as deployment-compatible.
3. This keeps one test suite valid across minimal and full-feature deployments.

## CI Mapping

Workflow: `.github/workflows/ci-tiered-tests.yml`

Jobs:
1. `unit-tier` -> `bash scripts/test_with_local_api.sh --suite unit`
2. `contract-local` -> `bash scripts/test_with_local_api.sh --suite contract`
3. `e2e-local` -> `bash scripts/test_with_local_api.sh --suite e2e`

Each local-API tier uploads `/tmp/cad_ml_uvicorn.log` for failure diagnosis.

## Marker Conventions

Registered in `pytest.ini`:
- `contract`
- `e2e`
- `perf`
- `performance`
- `slow`

Adding a new test class/module should include the appropriate marker so tier routing remains stable.
