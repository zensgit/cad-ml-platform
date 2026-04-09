# Benchmark Knowledge Outcome Drift CI Validation

## Scope
- Wire `benchmark_knowledge_outcome_drift` into `evaluation-report.yml`
- Propagate drift outputs into:
  - benchmark artifact bundle
  - benchmark companion summary
  - benchmark release decision
  - benchmark release runbook
- Expose drift status in job summary / downstream status surfaces
- Cover the workflow contract in unit tests

## Changed Files
- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Key Additions
- Added workflow dispatch inputs for:
  - `benchmark_knowledge_outcome_drift_enable`
  - `benchmark_knowledge_outcome_drift_current_summary_json`
  - `benchmark_knowledge_outcome_drift_previous_summary_json`
- Added downstream JSON passthrough for:
  - artifact bundle
  - companion summary
  - release decision
  - release runbook
- Added downstream output keys:
  - `knowledge_outcome_drift_status`
  - `knowledge_outcome_drift_summary`
  - `knowledge_outcome_drift_domain_regressions`
  - `knowledge_outcome_drift_domain_improvements`
  - `knowledge_outcome_drift_resolved_priority_domains`
  - `knowledge_outcome_drift_new_priority_domains`
  - `knowledge_outcome_drift_recommendations`
- Added job summary / signal-light rendering for standalone and downstream drift surfaces

## Validation
```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
python3 - <<'PY'
import yaml, pathlib
yaml.safe_load(pathlib.Path('.github/workflows/evaluation-report.yml').read_text())
print('yaml_ok')
PY
git diff --check
```

## Results
- `py_compile`: passed
- `flake8`: passed
- `pytest`: `3 passed, 1 warning`
- `yaml.safe_load(...)`: passed
- `git diff --check`: passed

## Notes
- This branch only wires CI / workflow contract.
- PR comment wiring should be stacked after this branch to keep workflow changes reviewable.
