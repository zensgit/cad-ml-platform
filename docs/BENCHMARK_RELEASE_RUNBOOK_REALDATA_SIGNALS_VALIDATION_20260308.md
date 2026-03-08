# Benchmark Release Runbook Real-Data Signals Validation

## Goal

Wire `benchmark_realdata_signals` into `export_benchmark_release_runbook.py` so the
operator-facing runbook includes real-data readiness, artifacts, and follow-up
recommendations.

## Changes

- Added `--benchmark-realdata-signals` CLI input.
- Added payload fields:
  - `realdata_status`
  - `realdata_signals`
  - `realdata_recommendations`
- Added `benchmark_realdata_signals` to the runbook artifact map.
- Added Markdown sections:
  - `## Real-Data Signals`
  - `## Real-Data Recommendations`
- Extended runbook review-signal derivation so partial real-data coverage flows
  into the operator action queue.

## Validation

Commands:

```bash
python3 -m py_compile scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_release_runbook.py
flake8 scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_release_runbook.py \
  --max-line-length=100
pytest -q tests/unit/test_benchmark_release_runbook.py
```

Results:

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `3 passed`

## Notes

- This keeps release runbook semantics aligned with companion summary and release
  decision outputs, which reduces benchmark operator drift across surfaces.
