# DEV_MAKE_VALIDATE_CORE_FAST_20260211

## Summary
- Added a single-command fast validation target for the current stable baseline.
- This target chains tolerance knowledge validation and service-mesh regression tests.

## Changes
- Updated `Makefile`
  - Added `.PHONY` entry: `validate-core-fast`
  - Added target: `make validate-core-fast`
    - `make validate-tolerance`
    - `make test-service-mesh`

## Validation
- `make validate-core-fast`
  - `validate-iso286`: `OK`
  - `test-tolerance`: `48 passed`
  - `test-service-mesh`: `103 passed`

