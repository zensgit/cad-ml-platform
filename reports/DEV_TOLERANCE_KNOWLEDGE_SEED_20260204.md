# DEV_TOLERANCE_KNOWLEDGE_SEED_20260204

## Summary
- Added precision/tolerance seed rules for GB/T 1804 and GB/T 1184 references.
- Added fundamental deviation lookup helper for hole/shaft symbols.
- Added unit test coverage for fundamental deviation lookup.

## Changes
- `data/knowledge/precision_rules.json`
  - Seed rules for general tolerance classes (GB/T 1804) and GD&T defaults (GB/T 1184).
- `src/core/knowledge/tolerance/fits.py`
  - Added `get_fundamental_deviation()` and fixed override loading indentation.
- `src/core/knowledge/tolerance/__init__.py`
  - Exported `get_fundamental_deviation`.
- `tests/unit/test_tolerance_fundamental_deviation.py`
  - Added tests for fundamental deviation lookup.

## Validation
- `pytest tests/unit/test_tolerance_fundamental_deviation.py -q`

## Notes
- Warning observed from `python_multipart` deprecation in Starlette (unrelated to tolerance logic).
