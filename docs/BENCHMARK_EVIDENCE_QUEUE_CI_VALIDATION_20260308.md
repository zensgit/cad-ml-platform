# Benchmark Evidence and Queue CI Validation 2026-03-08

## Goal

Wire the new benchmark scorecard dimensions into `evaluation-report.yml` so CI
can surface:

- assistant explainability readiness
- review queue readiness

This keeps the benchmark aligned with product-level competitiveness rather than
raw recognition accuracy only.

## Changes

- Added workflow-dispatch inputs:
  - `benchmark_scorecard_assistant_evidence_summary`
  - `benchmark_scorecard_review_queue_summary`
- Added env wiring:
  - `BENCHMARK_SCORECARD_ASSISTANT_EVIDENCE_SUMMARY_JSON`
  - `BENCHMARK_SCORECARD_REVIEW_QUEUE_SUMMARY_JSON`
- Extended benchmark scorecard step to forward the two JSON summaries when
  present.
- Exposed new benchmark outputs:
  - `assistant_status`
  - `review_queue_status`
- Extended job summary and PR comment with the new benchmark dimensions.

## Files

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation

Commands run:

```bash
python3 -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('.github/workflows/evaluation-report.yml').read_text())"
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Expected outcome

- workflow accepts the two new benchmark summary inputs
- benchmark step exports assistant/review-queue statuses
- job summary and PR comment surface both statuses

