# DEV_MAKE_VALIDATE_TOLERANCE_20260211

## Summary
- Added one-command tolerance stack validation for local development and regression checks.

## Changes
- Updated `Makefile`
  - Added `.PHONY` entry: `validate-tolerance`
  - Added target: `make validate-tolerance`
    - Executes `make validate-iso286`
    - Executes `make test-tolerance`

## Validation
- `make validate-tolerance`
  - `scripts/validate_iso286_deviations.py --spot-check`: `OK`
  - `scripts/validate_iso286_hole_deviations.py`: required symbols present
  - `make test-tolerance`: `44 passed`

