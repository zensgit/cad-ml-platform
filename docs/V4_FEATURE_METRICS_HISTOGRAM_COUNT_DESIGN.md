# V4 Feature Metrics Histogram Count Design

## Overview
Harden v4 surface_count and shape_entropy metric tests by asserting histogram
_count increments after extraction instead of relying on bucket samples.

## Updates
- Added histogram count helper for v4 metrics tests.
- Validated histogram count increments after v4 extraction.

## Files
- `tests/unit/test_v4_feature_performance.py`
