# DEV_FEATURE_EXTRACTOR_V4_REAL_20251224

## Scope
- Implement real v4 feature extractor tests (entropy, surface count, concurrency, upgrade/downgrade).

## Changes
- `tests/unit/test_feature_extractor_v4_real.py`
  - Replaced placeholder skips with functional assertions for v4 extraction behavior.

## Validation
- Command: `.venv/bin/python -m pytest tests/unit/test_feature_extractor_v4_real.py -v`
  - Result: 7 passed, 1 skipped.
