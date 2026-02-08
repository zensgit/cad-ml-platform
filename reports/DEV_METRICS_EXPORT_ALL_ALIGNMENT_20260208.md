# DEV_METRICS_EXPORT_ALL_ALIGNMENT_20260208

## Goal
Ensure `src/utils/analysis_metrics.py` remains internally consistent so tooling can validate observability guarantees:
- All defined metrics are exported via `__all__`
- `make verify-metrics` works as intended

## Changes
- `src/utils/analysis_metrics.py`
  - Updated `__all__` to include all metrics defined via `Counter(...)`, `Histogram(...)`, and `Gauge(...)`.
  - This includes the newly added analyze PartClassifier shadow metrics and previously un-exported `classification_cache_*` and `v16_*` metrics.

## Verification
Commands:
```bash
.venv/bin/python scripts/check_metrics_consistency.py
.venv/bin/python scripts/verify_metrics_export.py
make lint
.venv/bin/python -m mypy src
```

Result:
- Metrics consistency check passed.
- Metrics export verification passed.
- Lint + mypy passed locally.

