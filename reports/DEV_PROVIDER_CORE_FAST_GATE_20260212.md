# DEV_PROVIDER_CORE_FAST_GATE_20260212

## Summary
- Added a dedicated provider regression gate and integrated it into `validate-core-fast`.

## Changes
- Updated `Makefile`:
  - Added `test-provider-core` target to run provider framework critical tests.
  - Updated `validate-core-fast` to include `test-provider-core` after tolerance and service-mesh gates.

## Validation
- `make test-provider-core`
  - `57 passed`
- `make validate-core-fast`
  - `validate-iso286`: `OK`
  - `test-tolerance`: `48 passed`
  - `test-service-mesh`: `103 passed`
  - `test-provider-core`: `57 passed`

## Notes
- This keeps provider framework regressions in the same fast gate path as tolerance and service-mesh.
