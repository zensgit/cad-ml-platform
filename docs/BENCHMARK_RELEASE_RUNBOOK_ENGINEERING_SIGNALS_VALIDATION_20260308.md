# Benchmark Release Runbook Engineering Signals Validation

## Goal

Expose engineering signals to the benchmark release runbook so operator actions
and missing-artifact checks can account for engineering readiness alongside the
existing benchmark decision artifacts.

## Scope

- `scripts/export_benchmark_release_runbook.py`
- `tests/unit/test_benchmark_release_runbook.py`

## Delivered

- Added `--benchmark-engineering-signals` CLI input
- Added `benchmark_engineering_signals` artifact row
- Included engineering recommendations when engineering status is not ready
- Included engineering artifact in missing-artifact runbook logic

## Validation

```bash
python3 -m py_compile \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_release_runbook.py

flake8 \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_release_runbook.py \
  --max-line-length=100

pytest -q tests/unit/test_benchmark_release_runbook.py
```

## Result

- `py_compile`: pass
- `flake8`: pass
- `pytest`: pass
