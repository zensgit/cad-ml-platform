# Benchmark Release Decision Validation

## Scope

Adds a standalone `benchmark release decision` exporter that reduces the current
benchmark stack into one operator-facing release state:

- `ready`
- `review_required`
- `blocked`

## Inputs

- `benchmark_scorecard`
- `benchmark_operational_summary`
- `benchmark_artifact_bundle`
- `benchmark_companion_summary`

## Outputs

- `release_status`
- `automation_ready`
- `primary_signal_source`
- `blocking_signals`
- `review_signals`
- `component_statuses`
- Markdown + JSON artifacts

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

- standalone exporter added
- blocker-driven `blocked` state verified
- clean-path `ready` state verified
- CLI output generation verified
