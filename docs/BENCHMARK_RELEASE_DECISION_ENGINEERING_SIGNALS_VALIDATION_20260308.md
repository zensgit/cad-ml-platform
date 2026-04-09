# Benchmark Release Decision Engineering Signals Validation

## Goal

Extend benchmark release decision outputs so engineering signals become part of
the stable decision contract and artifact set.

## Scope

- `scripts/export_benchmark_release_decision.py`
- `tests/unit/test_benchmark_release_decision.py`

## Delivered

- Added `--benchmark-engineering-signals` CLI input
- Added `benchmark_engineering_signals` artifact row
- Added `engineering_signals` component status to release decision output
- Allowed engineering recommendations to participate in review signals

## Validation

```bash
python3 -m py_compile \
  scripts/export_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_decision.py

flake8 \
  scripts/export_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_decision.py \
  --max-line-length=100

pytest -q tests/unit/test_benchmark_release_decision.py
```

## Result

- `py_compile`: pass
- `flake8`: pass
- `pytest`: pass
