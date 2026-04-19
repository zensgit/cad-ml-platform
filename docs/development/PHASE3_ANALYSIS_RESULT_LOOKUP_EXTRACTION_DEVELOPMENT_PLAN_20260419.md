# Phase 3 Analyze Result Lookup Extraction Development Plan

## Goal
- Extract the `get_analysis_result` cache/load wrapper from `src/api/v1/analyze.py`.
- Keep route behavior and existing store/cache integrations unchanged.

## Scope
- Add `src/core/analysis_result_lookup.py`
- Update `src/api/v1/analyze.py` to delegate cached lookup
- Add focused unit coverage for the helper

## Constraints
- Preserve route path and 404 behavior
- Do not move the real cache/store implementations
- Keep cache key format and default TTL unchanged

## Validation Plan
- `py_compile` on touched files
- `flake8` on touched files
- Focused `pytest` for the new helper plus existing store coverage
