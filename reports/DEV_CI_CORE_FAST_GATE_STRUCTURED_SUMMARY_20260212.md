# DEV_CI_CORE_FAST_GATE_STRUCTURED_SUMMARY_20260212

## Summary
- Refined core-fast-gate `GITHUB_STEP_SUMMARY` output in both CI workflows to a structured markdown table.
- Removed dependency on `rg` in summary extraction logic to avoid runner-tooling variance; now uses `grep`, `sed`, and `tail` only.

## Changes
- Updated `.github/workflows/ci-tiered-tests.yml`
  - summary step now checks log existence early and exits cleanly when missing
  - computes check status for ISO286 deviations / hole symbols / tolerance suite / service-mesh suite
  - renders a markdown table with status and evidence
  - appends tail log block for diagnostics
- Updated `.github/workflows/ci.yml`
  - same structured summary behavior for the `tests` job core-fast-gate path
  - same shell-tool fallback (`grep`/`sed`/`tail`) to avoid `rg` coupling

## Validation
- `make validate-core-fast`
  - `validate-iso286`: `OK`
  - `test-tolerance`: `48 passed`
  - `test-service-mesh`: `103 passed`

## Notes
- This change is presentation/observability focused and does not alter gate criteria.
- The status-table logic is intentionally tolerant (`N/A`) if pass-line extraction fails, while preserving full tail logs for debugging.
