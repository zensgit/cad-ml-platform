# Benchmark Competitive Surpass Index Release Surfaces CI Validation

## Scope

- Wire `competitive_surpass_index` into benchmark release decision and release
  runbook workflow inputs.
- Export release-surface outputs for:
  - `competitive_surpass_index_status`
  - `competitive_surpass_primary_gaps`
  - `competitive_surpass_recommendations`
- Expose those values in:
  - evaluation job summary
  - PR comment
  - signal lights

## Changed Files

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation

```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Result

- `release decision` CI wiring now consumes and reports competitive surpass
  release-surface status.
- `release runbook` CI wiring now consumes and reports competitive surpass
  release-surface status.
- Job summary and PR comment contain dedicated rows for:
  - `Benchmark Release Decision Competitive Surpass`
  - `Benchmark Release Runbook Competitive Surpass`
