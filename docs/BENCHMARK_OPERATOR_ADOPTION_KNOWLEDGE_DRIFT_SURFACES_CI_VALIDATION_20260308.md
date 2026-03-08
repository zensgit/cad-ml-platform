# Benchmark Operator Adoption Knowledge Drift Surfaces CI Validation

Date: 2026-03-08
Branch: `feat/benchmark-operator-adoption-drift-surfaces-ci`

## Scope
- Expose operator adoption knowledge drift status and summary for:
  - benchmark artifact bundle
  - benchmark companion summary
  - benchmark release decision
  - benchmark release runbook
- Wire the new fields into `evaluation-report.yml` outputs, job summary, and PR comment
- Extend workflow regression coverage

## Files
- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation
- YAML parse: passed
- `flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100`: passed
- `pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`: passed

## Result
- Benchmark surfaces now expose operator adoption knowledge drift at the CI summary and PR comment level instead of only inside exporter JSON payloads.
