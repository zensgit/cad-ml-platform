# Feature Cache Tuning Metrics Contract Design

## Overview
Extend metrics contract coverage to include the cache tuning request counter and
register the metric during contract setup so label schemas are validated.

## Updates
- Added cache tuning request and recommendation gauges to the conditional metrics list:
  `feature_cache_tuning_requests_total{status}`,
  `feature_cache_tuning_recommended_capacity`,
  `feature_cache_tuning_recommended_ttl_seconds`.
- Triggered cache tuning endpoint during metrics registration to ensure the
  metrics appear for schema validation when metrics are enabled.

## Files
- `tests/test_metrics_contract.py`
