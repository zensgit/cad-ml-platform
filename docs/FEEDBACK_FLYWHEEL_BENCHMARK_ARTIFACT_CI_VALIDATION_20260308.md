# Feedback Flywheel Benchmark Artifact CI Validation

## Goal

Add a standalone feedback flywheel benchmark artifact to `evaluation-report.yml`, instead of
only exposing its status through the full benchmark scorecard.

## Design

- Reuse the existing benchmark feedback, fine-tune, and metric-train summary inputs.
- Generate JSON and Markdown artifacts with `scripts/export_feedback_flywheel_benchmark.py`.
- Upload the artifact independently from the scorecard artifact.
- Surface the artifact status and key counts in the GitHub job summary.

## Files

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation

```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Result

CI now emits a dedicated `feedback-flywheel-benchmark` artifact and prints its status,
feedback count, correction count, fine-tune sample count, and metric triplet count in the
job summary.
