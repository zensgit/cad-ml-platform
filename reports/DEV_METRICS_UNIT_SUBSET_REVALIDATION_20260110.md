# DEV_METRICS_UNIT_SUBSET_REVALIDATION_20260110

## Scope
Re-run metrics unit subset for cache tuning, opcode mode, and v4 feature performance.

## Command
- `pytest tests/unit/test_cache_tuning.py tests/unit/test_model_opcode_modes.py tests/unit/test_v4_feature_performance.py -v`

## Results
- 39 passed, 4 skipped (metrics counters unavailable).
