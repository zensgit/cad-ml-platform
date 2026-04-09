# Benchmark Companion Real-Data Signals Validation

## Scope

Wire `benchmark real-data signals` into the benchmark companion summary so the
operational surface exposes real DXF/`.h5`/STEP validation readiness.

## Delivered

- `scripts/export_benchmark_companion_summary.py`
- `tests/unit/test_benchmark_companion_summary.py`

The companion summary now emits:

- `realdata_status`
- `component_statuses.realdata_signals`
- `realdata_signals`
- `realdata_recommendations`
- `artifacts.benchmark_realdata_signals`

## Validation

```bash
python3 -m py_compile \
  scripts/export_benchmark_companion_summary.py \
  tests/unit/test_benchmark_companion_summary.py

flake8 \
  scripts/export_benchmark_companion_summary.py \
  tests/unit/test_benchmark_companion_summary.py \
  --max-line-length=100

pytest -q tests/unit/test_benchmark_companion_summary.py
```

## Result

- `py_compile`: passed
- `flake8`: passed
- `pytest`: passed
