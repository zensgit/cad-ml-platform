# Metrics Consistency Report

## Scope
- Verify `src/utils/analysis_metrics.py` metric definitions are exported via `__all__`.

## Test Run
- Command: `.venv/bin/python scripts/check_metrics_consistency.py`
- Result: `All 92 metrics are properly exported`

## Notes
- Dedup2d metrics are validated via analysis metrics export list; no missing or extra entries reported.
