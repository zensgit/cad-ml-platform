## Goal

Surface the benchmark operational summary in the PR comment emitted by
`evaluation-report.yml`, so reviewers can see operational status, blockers, and recommended
actions without downloading artifacts first.

## Files

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Design

- Reuse `steps.benchmark_operational_summary.outputs.*` from the CI export step.
- Add stable JS bindings for operational summary status, blockers, recommendations, and artifact.
- Render one row in the main PR comment analysis table and one signal-light row in the compact
  health table.
- Extend workflow regression tests so the comment contract does not silently regress.

## Validation

```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Result

Local workflow regression passed with the benchmark operational summary exposed in both the PR
comment analysis table and the signal-light section.
