# Feedback Flywheel Benchmark PR Comment Validation

## Goal

Expose the standalone feedback flywheel benchmark artifact in the PR comment, not just in the
job summary and uploaded artifacts.

## Design

- Reuse the existing `feedback_flywheel_benchmark` step outputs from `evaluation-report.yml`.
- Add a compact PR-comment line with:
  - status
  - feedback count
  - correction count
  - fine-tune sample count
  - metric triplet count
  - Markdown artifact path

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

The PR comment now shows the standalone feedback flywheel benchmark artifact alongside the
benchmark scorecard, assistant evidence report, and active-learning review queue summaries.
