# Benchmark Companion Engineering Signals Validation

## Goal

Extend benchmark companion summary outputs so they can consume the standalone
engineering signals artifact and expose it as a stable component in downstream
benchmark reporting.

## Scope

- `scripts/export_benchmark_companion_summary.py`
- `tests/unit/test_benchmark_companion_summary.py`

## Delivered

- Added `--benchmark-engineering-signals` CLI input
- Added `benchmark_engineering_signals` artifact row
- Added `engineering_signals` component status to companion summary output
- Included engineering signals in review surface readiness decisions
- Allowed engineering recommendations to feed companion recommended actions

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

- `py_compile`: pass
- `flake8`: pass
- `pytest`: pass
