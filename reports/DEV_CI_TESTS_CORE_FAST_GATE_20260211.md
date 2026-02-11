# DEV_CI_TESTS_CORE_FAST_GATE_20260211

## Summary
- Added core fast validation gate to `ci.yml` tests matrix for Python `3.11`.
- This aligns the primary CI workflow with the tiered workflow fast-gate strategy.

## Changes
- Updated `.github/workflows/ci.yml`
  - In `tests` job, added:
    - `Run core fast gate (3.11 only)`
    - `if: matrix.python-version == '3.11'`
    - `run: make validate-core-fast`
  - Step placement: after dependency installation, before quick smoke and larger test suites.

## Validation
- `make validate-core-fast`
  - `validate-iso286`: `OK`
  - `test-tolerance`: `48 passed`
  - `test-service-mesh`: `103 passed`

