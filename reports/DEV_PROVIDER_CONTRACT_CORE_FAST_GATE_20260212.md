# DEV_PROVIDER_CONTRACT_CORE_FAST_GATE_20260212

## Summary
- Added provider API contract regression suite into `validate-core-fast`.
- Extended CI core-fast summaries to include `provider-contract suite` row.

## Changes
- `Makefile`:
  - added target `test-provider-contract`.
  - integrated `test-provider-contract` into `validate-core-fast`.
- `.github/workflows/ci.yml`:
  - core-fast summary parser now reads a 5th `passed` line.
  - added summary row: `provider-contract suite`.
- `.github/workflows/ci-tiered-tests.yml`:
  - same parser/summary enhancement.

## Validation
- `make test-provider-contract`
  - result: `4 passed, 20 deselected`
- `make validate-core-fast`
  - result: `ISO286 validators OK`, `48 passed`, `3 passed`, `103 passed`, `59 passed`, `4 passed`

## Notes
- Provider contract gate currently covers:
  - provider health payload shape
  - core provider plugin summary shape
  - provider health OpenAPI schema diagnostics
  - health OpenAPI plugin summary schema
