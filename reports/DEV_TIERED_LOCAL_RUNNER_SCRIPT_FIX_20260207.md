# DEV_TIERED_LOCAL_RUNNER_SCRIPT_FIX_20260207

## Summary

Aligned `scripts/test_with_local_api.sh` with current `Makefile` + CI tiered workflow contract.

## Problem

- CI and local targets reference:
  - `--suite unit|contract|e2e|all`
  - `--wait-seconds`
- Existing local script (untracked) only supported:
  - `--suite all|smoke|unit|integration`
  - `--wait`
- This created a mismatch between documented/declared workflow and actual runner behavior.

## Changes

- Reworked `scripts/test_with_local_api.sh` to support:
  - `--suite unit|contract|e2e|all`
  - `--wait-seconds` (and backward-compatible `--wait` alias)
  - `--log-path`, `--python-bin`, `--pytest-bin`
- Added local API orchestration for API-backed suites:
  - Reuse healthy API if `API_BASE_URL` is already available.
  - Auto-start `uvicorn src.main:app` when targeting local base URL and API is not healthy.
  - Wait for `/health`, then run tests.
  - Always cleanup spawned uvicorn process via trap.
- Kept `unit` suite API-free by default.

## Validation

Commands executed:

```bash
bash -n scripts/test_with_local_api.sh
bash scripts/test_with_local_api.sh --help
bash scripts/test_with_local_api.sh --suite unit --pytest-bin echo
bash scripts/test_with_local_api.sh --suite contract --wait-seconds 45 --pytest-bin echo
bash scripts/test_with_local_api.sh --suite contract --wait-seconds 90
```

Results:

- Syntax check: PASS
- Help/options check: PASS
- `unit` route mapping check: PASS
- API orchestration dry run (`contract` + `echo`): PASS
- Real `contract` suite execution: PASS
  - `9 passed, 4 skipped` in `tests/contract`
- Post-run process cleanup check (`uvicorn src.main:app`): PASS

## Notes

- `src/core/providers/` remains untracked and incomplete (not modified by this fix).
- This report only covers the tiered local runner script alignment.
