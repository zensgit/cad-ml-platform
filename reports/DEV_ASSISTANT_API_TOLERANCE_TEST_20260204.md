# DEV_ASSISTANT_API_TOLERANCE_TEST_20260204

## Summary
- Added assistant API tolerance query coverage and guarded TestClient creation for local httpx incompatibility.

## Changes
- `tests/unit/assistant/test_llm_api.py`
  - Added `test_query_tolerance_precision` for GB/T 1804 queries.
  - Skipped API tests when TestClient is unavailable due to httpx/starlette mismatch.

## Validation
- `pytest tests/unit/assistant/test_llm_api.py -q`

## Notes
- Tests are skipped only when TestClient construction fails in the local environment.
