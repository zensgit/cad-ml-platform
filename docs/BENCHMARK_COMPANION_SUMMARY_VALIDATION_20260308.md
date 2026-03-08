# Benchmark Companion Summary Validation

## Goal

Add a standalone operator-facing benchmark companion summary that sits above the
existing benchmark scorecard, operational summary, and artifact bundle.

## Design

The companion summary is a lightweight roll-up for reviewers and operators. It:

- prefers the artifact bundle as the top-level benchmark state
- falls back to operational summary, then scorecard
- exposes:
  - `overall_status`
  - `review_surface`
  - `primary_gap`
  - `component_statuses`
  - `recommended_actions`
  - `blockers`
  - `artifacts`

This does not replace scorecard or bundle. It acts as a concise operational
entry point.

## Files

- `scripts/export_benchmark_companion_summary.py`
- `tests/unit/test_benchmark_companion_summary.py`

## Validation

Commands run:

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

## Expected result

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `3 passed`
