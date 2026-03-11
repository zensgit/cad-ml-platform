# Benchmark Knowledge Domain Release Readiness Action Plan CI Validation

## Scope
- add workflow-dispatch inputs and environment defaults for
  `benchmark_knowledge_domain_release_readiness_action_plan`
- build the standalone exporter in `evaluation-report.yml`
- upload the generated artifact and expose summary lines in the job summary

## Key Files
- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation
- `pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
- `python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
- `python3 - <<'PY' ... yaml.safe_load('.github/workflows/evaluation-report.yml') ... PY`

## Result
- workflow contract regression passed
- workflow YAML parsed successfully
- CI wiring now covers:
  - build
  - artifact upload
  - job summary
