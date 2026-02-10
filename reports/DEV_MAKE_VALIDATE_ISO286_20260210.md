# DEV_MAKE_VALIDATE_ISO286_20260210

## Summary
- Added a lightweight Makefile target to validate the ISO286/GB-T 1800 deviation table artifacts without running the full test suite.

## Change
- Added `make validate-iso286` to `Makefile`.

## Validation
- `make validate-iso286`
  - Runs:
    - `python3 scripts/validate_iso286_deviations.py --spot-check`
    - `python3 scripts/validate_iso286_hole_deviations.py`
  - Result: pass

