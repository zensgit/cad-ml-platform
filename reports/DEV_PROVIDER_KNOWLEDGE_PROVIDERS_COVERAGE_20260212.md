# DEV_PROVIDER_KNOWLEDGE_PROVIDERS_COVERAGE_20260212

## Goal
Increase provider framework completeness by validating that the built-in
knowledge providers (`knowledge/tolerance`, `knowledge/standards`) are not just
registered, but also pass health checks and return deterministic payloads.

## Change
- Added unit coverage for knowledge providers:
  - `knowledge/tolerance`: health + process payload counts
  - `knowledge/standards`: health + process payload counts
- Wired the new test into the provider-core regression target so it runs under
  `make validate-core-fast`.

## Implementation
- `tests/unit/test_provider_knowledge_providers.py`
  - Async tests that bootstrap registry, run `health_check(timeout_seconds=0.5)`,
    then assert `process()` returns expected counts.
- `Makefile`
  - Added the test file to `test-provider-core`.

## Validation
- `.venv/bin/python -m pytest tests/unit/test_provider_knowledge_providers.py -v`
  - Result: `2 passed`
- `make validate-core-fast`
  - Result: passed

## Files Changed
- `tests/unit/test_provider_knowledge_providers.py`
- `Makefile`

