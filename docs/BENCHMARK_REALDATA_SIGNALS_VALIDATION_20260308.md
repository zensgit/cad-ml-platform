# Benchmark Real-Data Signals Validation

## Scope

Add a reusable benchmark real-data exporter that summarizes:

- DXF hybrid real-data validation
- online `.h5` example validation
- STEP smoke validation
- STEP directory validation

## Delivered

- `src/core/benchmark/realdata_signals.py`
- `scripts/export_benchmark_realdata_signals.py`
- `tests/unit/test_benchmark_realdata_signals.py`

The exporter now emits a stable benchmark payload with:

- `status`
- `component_statuses`
- `components.hybrid_dxf`
- `components.history_h5`
- `components.step_smoke`
- `components.step_dir`
- `recommendations`

## Validation

```bash
python3 -m py_compile \
  src/core/benchmark/realdata_signals.py \
  scripts/export_benchmark_realdata_signals.py \
  tests/unit/test_benchmark_realdata_signals.py

flake8 \
  src/core/benchmark/realdata_signals.py \
  scripts/export_benchmark_realdata_signals.py \
  tests/unit/test_benchmark_realdata_signals.py \
  --max-line-length=100

pytest -q tests/unit/test_benchmark_realdata_signals.py
```

## Result

- `py_compile`: passed
- `flake8`: passed
- `pytest`: `4 passed`
