# Vector Migrate Dimension Histogram Count Design

## Overview
Confirm the vector migration dimension delta histogram records observations by
asserting the count increases after a migration run.

## Updates
- Added a histogram count assertion guarded by a Prometheus availability check.
- Reused an existing v1 vector to trigger a v4 migration observation.

## Files
- `tests/unit/test_vector_migrate_dimension_histogram.py`
