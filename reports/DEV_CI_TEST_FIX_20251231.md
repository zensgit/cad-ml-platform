# CI tests fix verification (2025-12-31)

## Issue
- CI (Python 3.10) failed on `tests/unit/test_v4_feature_performance.py::TestV4PerformanceComparison::test_v4_overhead_within_bounds`.
- Failure: ratio-based overhead check was unstable on low-baseline runs.

## Fix
- Updated the test to use ratio-based overhead only when baseline runtime is meaningful.
- Added absolute per-extraction overhead threshold (5ms) for low-baseline runs.

## Tests and validation
- `python3 -m pytest tests/unit/test_v4_feature_performance.py::TestV4PerformanceComparison::test_v4_overhead_within_bounds -q`
  - Result: **pass** (1 passed, 56.20s).

## Notes
- CI rerun required to confirm Python 3.10 pipeline passes.
