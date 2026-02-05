# DEV_CLASSIFIER_RECORDING_RULE_PREFIX_20260205

## Summary
Renamed the classifier cache hit ratio recording rule to include a standard
`cad_ml_` prefix so promtool naming validation passes without adding new
prefix warnings.

## Changes
- Updated `docs/prometheus/recording_rules.yml`:
  - `classification_cache_hit_ratio` → `cad_ml_classification_cache_hit_ratio`

## Notes
- Existing prefix warnings (`top_rejection_reason_rate`, `memory_exhaustion_rate`) remain unchanged.

## Verification
- `python3 scripts/validate_prom_rules.py`
