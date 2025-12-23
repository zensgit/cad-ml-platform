# DEV_WEEK7_SHAPE_ENTROPY_PERF_OPTIMIZATION_20251223

## Context
- CI failed on `test_shape_entropy_performance` due to a 1s threshold overrun.

## Changes
- `src/core/feature_extractor.py`: compute shape entropy in a single pass to reduce per-call overhead.

## Tests
- `pytest tests/unit/test_v4_feature_performance.py::TestV4PerformanceComparison::test_shape_entropy_performance -q`
- `pytest tests/unit/test_v4_feature_performance.py::TestV4PerformanceComparison::test_v4_overhead_within_bounds -q`

## Results
- Targeted performance tests passed locally.
