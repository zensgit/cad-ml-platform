# DEV_DEDUP2D_STORAGE_METRICS_20260101

## Design Summary
- Added dedup2d file storage observability to both local and S3 backends.
- Metrics covered: upload/download/delete counters, upload byte histogram, and operation duration histogram.
- Metrics increments are wrapped with `safe_inc`/`safe_observe` to avoid impacting storage flow on metric failures.

## Code Changes
- `src/core/dedup2d_file_storage.py`
  - instrumented `save_bytes`, `load_bytes`, `delete` for local and S3 backends
  - added duration + size tracking for uploads
- `tests/unit/test_dedup2d_file_storage_metrics.py`
  - new unit test to assert metric counters increment on local storage operations

## Verification
```bash
pytest tests/unit/test_dedup2d_file_storage_metrics.py -v
```
Result:
- `1 skipped` (`prometheus_client` not available in current test runtime)

## Notes
- When Prometheus client is present, the unit test will validate counter increments.
