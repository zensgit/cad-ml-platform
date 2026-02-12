# DEV_PROVIDER_FRAMEWORK_CLARITY_20260208

Date: 2026-02-08

## Summary

Clarified that `src/core/providers/` is a first-class provider framework (not an incomplete/temporary module) and documented how to extend it safely. Also clarified the canonical local tiered test runner script location (`scripts/test_with_local_api.sh`).

## Changes

- Updated provider framework documentation:
  - `docs/PROVIDER_FRAMEWORK.md`
- Added an explicit status note to the historical architecture analysis so it does not read as an unimplemented plan:
  - `claudedocs/ARCHITECTURE_ANALYSIS_WEEK2.md`
- Clarified that the canonical local API test runner is `scripts/test_with_local_api.sh` (used by `Makefile` and CI), and that a repo-root `test_with_local_api.sh` should be avoided:
  - `docs/TESTING_STRATEGY.md`

## Validation

- Shell script syntax check:
  - `bash -n scripts/test_with_local_api.sh`
- Provider framework unit suite:
  - `pytest tests/unit/test_provider_framework.py -q`
  - Result: `10 passed`
