# Benchmark Operator Adoption Knowledge Drift CI Validation

Date: 2026-03-08
Branch: `feat/benchmark-operator-adoption-drift-ci`

## Scope
- Wire `benchmark_operator_adoption_knowledge_drift_json` into `evaluation-report.yml`
- Propagate operator adoption knowledge drift into workflow outputs, job summary, and PR comment
- Extend workflow regression tests for the new contract

## Files
- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation
- YAML parse: passed
- `flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100`: passed
- `pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`: passed

## Result
- Operator adoption now consumes knowledge drift in CI dispatch inputs, build-step wiring, summary output, and PR comment rendering.
