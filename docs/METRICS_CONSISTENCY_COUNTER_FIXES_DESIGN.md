# Metrics Consistency and Counter Fixes Design

## Scope
- Ensure analysis result store cleanup metrics are exported for consistency checks.
- Stabilize vector migrate metrics counters in tests by ignoring `_created` samples.
- Tighten typing in analysis result store for mypy compliance.
- Keep examples and comments within lint line-length limits.

## Problem Statement
- `scripts/check_metrics_consistency.py` failed because cleanup metrics were defined but missing from `__all__`.
- `test_vector_migrate_metrics_downgraded_status` could read the `_created` sample instead of the counter value.
- `mypy` flagged untyped dict returns and ambiguous `status` values passed to metrics.

## Design
- Export `analysis_result_cleanup_total`, `analysis_result_cleanup_deleted_total`, and
  `analysis_result_store_files` via `__all__` in `src/utils/analysis_metrics.py`.
- Ignore samples ending with `_created` when collecting counter values in
  `tests/unit/test_vector_migrate_metrics.py`.
- Add explicit type hints and casts in `src/utils/analysis_result_store.py` so JSON loads
  are treated as `dict[str, Any]`, and use explicit `status` variables for metrics.
- Wrap long JSON example lines and comments to satisfy `flake8` line-length checks.

## Impact
- No API changes; metrics consistency checks now pass and tests are stable.
- Cleanup metrics are available to downstream importers.

## Validation
- `python3 scripts/check_metrics_consistency.py`
- `python3 -m pytest tests/unit/test_vector_migrate_metrics.py -k downgraded -v`
- `python3 -m mypy src/utils/analysis_result_store.py`
- `python3 -m flake8 src`
