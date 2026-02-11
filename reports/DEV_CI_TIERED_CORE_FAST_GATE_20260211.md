# DEV_CI_TIERED_CORE_FAST_GATE_20260211

## Summary
- Integrated the local stable validation baseline into CI tiered workflow as a pre-gate.
- This catches tolerance/service-mesh regressions before unit/contract/e2e tiers start.

## Changes
- Updated `.github/workflows/ci-tiered-tests.yml`
  - Added job: `core-fast-gate`
  - Added step: `make validate-core-fast`
  - Updated dependency chain: `unit-tier` now `needs: [core-fast-gate]`

## Validation
- `make validate-core-fast`
  - `validate-iso286`: `OK`
  - `test-tolerance`: `48 passed`
  - `test-service-mesh`: `103 passed`

