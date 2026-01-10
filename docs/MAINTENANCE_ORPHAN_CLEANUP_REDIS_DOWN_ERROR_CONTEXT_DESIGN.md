# Maintenance Orphan Cleanup Redis Down Error Context Design

## Overview
Harden orphan cleanup error responses when Redis is unavailable and ensure the
orphan cleanup metric increments on failure.

## Updates
- Added `operation` and `resource_id` context fields to orphan cleanup errors.
- Incremented `vector_orphan_total` on Redis instability aborts.
- Extended unit tests to validate error context and metric increments.

## Files
- `src/api/v1/maintenance.py`
- `tests/unit/test_orphan_cleanup_redis_down.py`
