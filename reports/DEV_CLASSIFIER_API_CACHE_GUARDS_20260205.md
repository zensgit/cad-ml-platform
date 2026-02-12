# DEV_CLASSIFIER_API_CACHE_GUARDS_20260205

## Summary
Added Prometheus metrics for classifier cache usage, guarded cache endpoints with
admin token auth, and introduced a regression test for cache hits.

## Changes
- Metrics: added cache hit/miss counters and cache size gauge in
  `src/utils/analysis_metrics.py` and wired them into `src/inference/classifier_api.py`.
- Auth: `/cache/stats` and `/cache/clear` now require `X-Admin-Token`.
- Test: `tests/unit/test_classifier_api_cache.py` validates cache hit behavior.

## Tests
- `python3 -m pytest tests/unit/test_classifier_api_cache.py -q`

## Notes
- FastAPI emits a deprecation warning for `@app.on_event`; consider migrating
  to lifespan handlers when convenient.
