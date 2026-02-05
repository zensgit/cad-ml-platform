# DEV_CLASSIFIER_RATE_LIMIT_DOCS_20260205

## Summary
Documented classifier rate-limit environment variables and added classification
cache metrics to the metrics index and recording rules.

## Changes
- Added `CLASSIFIER_RATE_LIMIT_PER_MIN` and `CLASSIFIER_RATE_LIMIT_BURST` to
  `docs/DEPLOYMENT.md`.
- Listed classification cache metrics in `docs/METRICS_INDEX.md`.
- Added `cad_ml_classification_cache_hit_ratio` recording rule in
  `docs/prometheus/recording_rules.yml`.

## Notes
- Recording rules update is documentation-only; update production Prometheus
  configuration if you want the new record in dashboards.
