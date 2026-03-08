## Goal

Wire the standalone benchmark operational summary into `evaluation-report.yml` so CI can
generate, upload, and summarize the artifact alongside the existing benchmark scorecard and
feedback flywheel artifacts.

## Files

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Design

- Add workflow-dispatch inputs and env vars for optional benchmark operational summary sources.
- Generate the operational summary when either explicit enablement is set or at least one
  upstream benchmark artifact is available.
- Upload the JSON/Markdown pair as a dedicated artifact.
- Surface overall/component status, blockers, recommendations, and artifact path in the GitHub
  job summary.

## Validation

```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Result

Local workflow regression passed after adding operational summary env/input coverage, build step,
artifact upload, and job-summary assertions.
