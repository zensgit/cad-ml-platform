# Feature Cache Tuning POST Endpoint Design

## Overview
Add a POST cache tuning endpoint that accepts explicit hit rate and capacity
inputs to generate deterministic capacity/TTL recommendations, plus a request
counter metric.

## Updates
- Added `feature_cache_tuning_requests_total{status}` metric for request tracking.
- Implemented POST `/api/v1/features/cache/tuning` with request/response models.
- Added tuning heuristics for low, moderate, and high hit-rate bands.

## Files
- `src/utils/analysis_metrics.py`
- `src/api/v1/features.py`
- `tests/unit/test_cache_tuning.py`
