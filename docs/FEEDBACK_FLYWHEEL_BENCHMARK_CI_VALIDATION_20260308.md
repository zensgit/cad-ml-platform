# Feedback Flywheel Benchmark CI Validation

## Goal

Wire feedback flywheel benchmark summaries into `evaluation-report.yml` so the benchmark
scorecard can consume reviewed feedback, fine-tune, and metric-training artifacts in CI.

## Design

- Add optional `workflow_dispatch` inputs for:
  - `benchmark_scorecard_feedback_summary`
  - `benchmark_scorecard_finetune_summary`
  - `benchmark_scorecard_metric_train_summary`
- Add matching environment variables so scheduled or repository-level runs can inject the
  same summaries without editing the workflow.
- Extend the optional benchmark scorecard step so it passes the new summary paths to
  `scripts/generate_benchmark_scorecard.py`.
- Parse `feedback_flywheel.status` from the generated JSON and expose it as a step output.
- Surface the status in both:
  - the GitHub job summary
  - the PR comment benchmark table

## Implementation

Updated files:

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation

Commands:

```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Expected coverage:

- workflow env includes the new benchmark summary variables
- workflow dispatch exposes the new optional inputs
- benchmark step passes `--feedback-summary`, `--finetune-summary`, and
  `--metric-train-summary`
- benchmark outputs include `feedback_flywheel_status`
- GitHub summary prints `Benchmark feedback flywheel status`
- PR comment includes the feedback flywheel row and inline benchmark status

## Result

The benchmark CI path can now ingest feedback flywheel artifacts end-to-end and expose the
status in the same report surface as hybrid, history, B-Rep, governance, OCR, review queue,
and Qdrant readiness.
