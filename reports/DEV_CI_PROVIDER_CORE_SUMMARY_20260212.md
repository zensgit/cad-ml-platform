# DEV_CI_PROVIDER_CORE_SUMMARY_20260212

## Summary
- Extended CI core-fast-gate step summaries to include provider-core suite status/evidence.

## Changes
- Updated `.github/workflows/ci.yml` and `.github/workflows/ci-tiered-tests.yml`:
  - Parse third `passed in` line as provider-core evidence.
  - Add summary row:
    - `provider-core suite | ✅/❌ | <pass line>`
  - Keep shell tooling portable (`grep`/`sed`/`tail`).

## Validation
- Local dry-run equivalent via `make validate-core-fast` confirms pass-line output for all three suites.
- Summary script logic remains tolerant with `N/A` fallback when expected lines are missing.

## Notes
- Existing tolerance/service-mesh summary rows are unchanged.
