# Benchmark Artifact Bundle Engineering Signals Validation

## Goal

Expose benchmark engineering signals as a first-class artifact and component in
the compact artifact bundle so downstream release decision and runbook stages
can consume it without re-reading raw benchmark inputs.

## Scope

- `scripts/export_benchmark_artifact_bundle.py`
- `tests/unit/test_benchmark_artifact_bundle.py`

## Delivered

- Added `--benchmark-engineering-signals` CLI input
- Added `benchmark_engineering_signals` artifact row
- Added `engineering_signals` to bundle `component_statuses`
- Allowed engineering recommendations to participate in bundle recommendations

## Validation

```bash
python3 -m py_compile \
  scripts/export_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_artifact_bundle.py

flake8 \
  scripts/export_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  --max-line-length=100

pytest -q tests/unit/test_benchmark_artifact_bundle.py
```

## Result

- `py_compile`: pass
- `flake8`: pass
- `pytest`: pass
