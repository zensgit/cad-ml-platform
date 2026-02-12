# DEV_CLASSIFIER_CACHE_MONITORING_20260205

## Summary
Added basic rate limiting for classifier endpoints, emitted cache metrics, and
captured Prometheus panel/alert suggestions for cache behavior.

## Metrics
- `classification_cache_hits_total`
- `classification_cache_miss_total`
- `classification_cache_size`
- `classification_rate_limited_total`

## Suggested dashboards
- Cache hit rate:
  - `rate(classification_cache_hits_total[5m]) / (rate(classification_cache_hits_total[5m]) + rate(classification_cache_miss_total[5m]))`
- Cache size:
  - `classification_cache_size`
- Rate limiting:
  - `rate(classification_rate_limited_total[5m])`

## Suggested alerts
- **Low cache hit rate**: hit rate < 0.2 for 15m
- **Cache saturation**: cache size near configured max for 30m
- **Excess rate limiting**: rate limited > 5/min for 10m

## Tests
- `python3 -m pytest tests/unit/test_classifier_api_cache.py -q`
