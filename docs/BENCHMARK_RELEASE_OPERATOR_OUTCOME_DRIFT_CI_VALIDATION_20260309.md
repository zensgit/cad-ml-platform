# Benchmark Release Operator Outcome Drift CI Validation

## Goal
- Wire `operator_adoption_knowledge_outcome_drift` from benchmark release decision and release runbook into CI outputs, job summary, and PR comment surfaces.

## Scope
- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Changes
- Release decision step outputs now expose:
  - `operator_adoption_knowledge_outcome_drift_status`
  - `operator_adoption_knowledge_outcome_drift_summary`
- Release runbook step outputs now expose:
  - `operator_adoption_knowledge_outcome_drift_status`
  - `operator_adoption_knowledge_outcome_drift_summary`
- Job summary now includes:
  - `Benchmark release operator adoption knowledge outcome drift`
  - `Benchmark release runbook operator adoption knowledge outcome drift`
- PR comment / signal detail table now includes the same release-surface rows.
- Release decision and runbook status lines now inline operator outcome-drift detail.

## Validation
```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path('.github/workflows/evaluation-report.yml').read_text())
print('yaml_ok')
PY
git diff --check
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Result
- `py_compile`: passed
- `flake8`: passed
- workflow YAML parse: passed
- `git diff --check`: passed
- `pytest`: `3 passed, 1 warning`

## Outcome
- Release decision / runbook now surface operator-adoption outcome drift with stable CI outputs.
- Downstream PR comment consumers no longer need to infer operator outcome drift from free-form markdown.
