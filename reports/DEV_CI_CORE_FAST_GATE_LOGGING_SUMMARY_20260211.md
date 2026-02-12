# DEV_CI_CORE_FAST_GATE_LOGGING_SUMMARY_20260211

## Summary
- Improved observability for the new CI core-fast-gate checks in both main and tiered workflows.
- Added artifact upload and `GITHUB_STEP_SUMMARY` output so gate results are easy to inspect without opening raw logs.

## Changes
- Updated `.github/workflows/ci-tiered-tests.yml`
  - `core-fast-gate` now writes output to `/tmp/core-fast-gate-tiered.log` via `tee`
  - uploads artifact: `core-fast-gate-tiered-log`
  - appends key lines + tail to `GITHUB_STEP_SUMMARY`
- Updated `.github/workflows/ci.yml`
  - `tests` job (Python `3.11` only) gate now writes `/tmp/core-fast-gate-ci.log` via `tee`
  - uploads artifact: `core-fast-gate-ci-log-${{ matrix.python-version }}`
  - appends key lines + tail to `GITHUB_STEP_SUMMARY`

## Validation
- `make validate-core-fast`
  - `validate-iso286`: `OK`
  - `test-tolerance`: `48 passed`
  - `test-service-mesh`: `103 passed`

