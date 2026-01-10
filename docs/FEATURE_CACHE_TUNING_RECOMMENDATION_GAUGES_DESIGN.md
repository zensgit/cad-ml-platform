# Feature Cache Tuning Recommendation Gauges Design

## Overview
Expose the latest cache tuning recommendations as gauges so dashboards can
visualize recommended capacity and TTL alongside request counts.

## Updates
- Added `feature_cache_tuning_recommended_capacity` and
  `feature_cache_tuning_recommended_ttl_seconds` gauges.
- Updated POST and GET cache tuning endpoints to set the gauges.
- Aligned Grafana dashboard cache tuning panel to use the new metric names.
- Extended cache tuning tests to assert gauge updates when metrics are enabled.

## Files
- `src/utils/analysis_metrics.py`
- `src/api/v1/features.py`
- `src/api/v1/health.py`
- `tests/unit/test_cache_tuning.py`
- `config/grafana/dashboard_main.json`
