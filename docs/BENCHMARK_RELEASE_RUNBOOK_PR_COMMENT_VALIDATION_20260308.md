# Benchmark Release Runbook PR Comment Validation 2026-03-08

## Goal

Expose release runbook actionability directly in PR comments so reviewers can
see freeze readiness, next action, and missing artifacts without opening the
artifact bundle manually.

## Delivered

- PR comment now includes `Benchmark Release Runbook`
- signal-light table now includes runbook state
- duplicate benchmark artifact bundle / companion summary rows are removed from
  the PR comment tables

## Validation

```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
python3 - <<'PY'
from pathlib import Path
import yaml
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text(encoding='utf-8'))
print('yaml_ok')
PY
```
