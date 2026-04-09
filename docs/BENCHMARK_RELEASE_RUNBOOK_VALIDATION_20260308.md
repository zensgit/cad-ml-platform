# Benchmark Release Runbook Validation 2026-03-08

## Goal

Add an operator-facing release runbook exporter that turns benchmark release
decision artifacts into concrete next steps, so the platform compares on
engineering governance and release discipline instead of raw extraction only.

## Delivered

- new exporter:
  - `scripts/export_benchmark_release_runbook.py`
- new standalone JSON / Markdown output with:
  - `ready_to_freeze_baseline`
  - `missing_artifacts`
  - `blocking_signals`
  - `review_signals`
  - `next_action`
  - `operator_steps`
- release decision, companion summary, and artifact bundle are now translated
  into an operator checklist

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

## Expected Outcome

- blocked benchmark releases show missing artifacts and blocker resolution first
- review-required releases surface review steps before rerun/freeze
- ready releases produce a freeze-baseline step as the next action
