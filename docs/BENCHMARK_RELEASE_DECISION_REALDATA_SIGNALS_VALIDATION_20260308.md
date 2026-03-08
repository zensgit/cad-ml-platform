# Benchmark Release Decision Real-Data Signals Validation

## Goal

Wire `benchmark_realdata_signals` into `export_benchmark_release_decision.py` so the
release decision surface can expose real-data readiness alongside knowledge,
engineering, and operator-adoption inputs.

## Changes

- Added `--benchmark-realdata-signals` CLI input.
- Added `component_statuses.realdata_signals`.
- Added payload fields:
  - `realdata_status`
  - `realdata_signals`
  - `realdata_recommendations`
- Added `benchmark_realdata_signals` to artifact tracking.
- Added Markdown sections:
  - `## Real-Data Signals`
  - `## Real-Data Recommendations`
- Extended release-decision review signal derivation so partial/missing real-data
  coverage contributes operator-facing guidance.

## Validation

Commands:

```bash
python3 -m py_compile scripts/export_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_decision.py
flake8 scripts/export_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_decision.py \
  --max-line-length=100
pytest -q tests/unit/test_benchmark_release_decision.py
```

Results:

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `5 passed`

## Notes

- The release decision surface now preserves the same real-data signal semantics
  already introduced in the benchmark companion summary.
- This branch only changes the release-decision exporter and its tests.
