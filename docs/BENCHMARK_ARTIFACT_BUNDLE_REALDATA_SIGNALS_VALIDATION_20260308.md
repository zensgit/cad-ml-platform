# Benchmark Artifact Bundle Real-Data Signals Validation

## Goal

Expose `benchmark_realdata_signals` through the benchmark artifact bundle so the
bundle, downstream companion summary, and release surfaces can consume stable
real-data readiness evidence.

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

## Validation

Commands:

```bash
python3 -m py_compile scripts/export_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_artifact_bundle.py
flake8 scripts/export_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  --max-line-length=100
pytest -q tests/unit/test_benchmark_artifact_bundle.py
```

Results:

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `6 passed`

## Notes

- This branch keeps artifact-bundle semantics aligned with the already-added
  `realdata_signals` surface in the benchmark companion summary.
