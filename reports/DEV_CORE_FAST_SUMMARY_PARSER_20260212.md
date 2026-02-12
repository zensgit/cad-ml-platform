# DEV_CORE_FAST_SUMMARY_PARSER_20260212

## Summary
- Replaced brittle shell parsing of core-fast-gate logs in CI with a dedicated Python summarizer.
- Added unit coverage for log parsing behavior (complete log + missing suite fallback).

## Changes
- Added `scripts/ci/summarize_core_fast_gate.py`:
  - parses core-fast log by suite markers (`make test-*` / `make validate-openapi`) instead of fixed nth `passed` line assumptions.
  - outputs unified markdown summary table + tail section.
- Added `tests/unit/test_core_fast_gate_summary.py`:
  - validates full suite extraction for all core-fast suites.
  - validates `N/A` fallback behavior when suites are absent.
- Updated workflows to use the script:
  - `.github/workflows/ci.yml`
  - `.github/workflows/ci-tiered-tests.yml`

## Validation
- `pytest -q tests/unit/test_core_fast_gate_summary.py`
  - result: `2 passed`
- Dry-run behavior without log file:
  - `python3 scripts/ci/summarize_core_fast_gate.py --log-file /tmp/nonexistent.log --title "Core Fast Gate (Local Dry Run)"`
  - result: emits `No core-fast-gate log found.`
- Regression gate:
  - `make validate-core-fast`
  - result: `ISO286 validators OK`, `48 passed`, `3 passed`, `103 passed`, `59 passed`, `4 passed`

## Notes
- Core-fast summary generation is now centralized and easier to evolve when suite ordering changes.
