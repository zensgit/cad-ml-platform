# DEV_STABILITY_CI_TIERED_TESTING_20260206

## Summary

Implemented tiered test execution and CI alignment to make local/CI behavior reproducible:

1. Registered missing pytest markers to remove unknown-marker warnings.
2. Added a local API orchestration test runner script for `unit`, `contract`, `e2e`, and `all` suites.
3. Added Makefile tier targets that call the same script.
4. Added a dedicated GitHub Actions workflow for tiered jobs (`unit-tier`, `contract-local`, `e2e-local`).
5. Added/updated testing and CI triage docs.

## Code Changes

- `pytest.ini`
  - Added markers: `contract`, `e2e`, `perf`.
- `scripts/test_with_local_api.sh` (new)
  - Auto-starts/stops local API when needed.
  - Reuses an existing healthy API endpoint when available.
  - Supports `--suite unit|contract|e2e|all`, `--base-url`, `--api-key`, `--python`.
  - Uses `python3` by default to avoid stale/incomplete `.venv` mismatches.
- `Makefile`
  - Added targets:
    - `test-unit`
    - `test-contract-local`
    - `test-e2e-local`
    - `test-all-local`
  - `test-unit` now uses the same local runner script for interpreter consistency.
- `.github/workflows/ci-tiered-tests.yml` (new)
  - Added tiered jobs:
    - `unit-tier`
    - `contract-local`
    - `e2e-local`
  - Uploads `/tmp/cad_ml_uvicorn.log` artifact for API-backed tier failures.
- `docs/TESTING_STRATEGY.md` (new)
  - Documented tier definitions, commands, route-availability policy, and CI mapping.
- `docs/CI_FAILURE_ROUTING.md`
  - Added test-tier routing and triage procedure.

## Verification

Executed and passed:

1. `python3 -m pytest tests/unit -q -x`
   - `6964 passed, 36 skipped`
2. `bash scripts/test_with_local_api.sh --suite contract --wait-seconds 120`
   - `9 passed, 4 skipped`
3. `bash scripts/test_with_local_api.sh --suite e2e --wait-seconds 120`
   - `11 passed, 16 skipped`
4. `bash scripts/test_with_local_api.sh --suite all --wait-seconds 120`
   - `7515 passed, 72 skipped`
5. `make test-unit`
   - `6964 passed, 36 skipped`

## Notes

1. Existing deprecation warnings from `faiss` (`distutils` version classes) remain non-blocking.
2. Optional `/api/v2/*` endpoints are intentionally treated as deployment-optional in e2e tests (404 accepted where appropriate).
