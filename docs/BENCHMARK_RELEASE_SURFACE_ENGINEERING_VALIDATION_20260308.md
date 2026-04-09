## Scope

This change extends benchmark engineering-signal visibility across the release
surfaces:

- benchmark release decision
- benchmark release runbook
- benchmark companion summary PR comment surface
- benchmark artifact bundle PR comment surface

## Delivered

- release decision workflow inputs now accept engineering signals JSON
- release runbook workflow inputs now accept engineering signals JSON
- release decision workflow outputs now expose `engineering_status`
- release runbook workflow outputs now expose `engineering_status`
- step summary now includes release decision and runbook engineering status
- PR comment lines now include engineering status for:
  - artifact bundle
  - companion summary
  - release decision
  - release runbook

## Validation

```bash
python3 -m py_compile \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_release_runbook.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_release_runbook.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_release_runbook.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Expected result:

- exporter tests pass
- workflow contract tests pass
- workflow summary and PR comment coverage include engineering fields
