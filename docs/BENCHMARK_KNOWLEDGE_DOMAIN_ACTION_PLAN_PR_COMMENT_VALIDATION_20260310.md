# Benchmark Knowledge Domain Action Plan PR Comment Validation

Date: 2026-03-10

## Scope

This change extends the benchmark PR comment and signal lights to include
`knowledge_domain_action_plan` across:

- standalone benchmark action-plan status
- artifact bundle surface
- companion summary surface
- release decision surface
- release runbook surface

## Files

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation

```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
python3 - <<'PY'
from pathlib import Path
import yaml
yaml.safe_load(Path('.github/workflows/evaluation-report.yml').read_text())
print('yaml_ok')
PY
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Results:

- `py_compile` passed
- `flake8` passed
- workflow YAML parse passed
- `pytest` passed: `3 passed`

## Notes

- PR comment now renders:
  - `Benchmark Knowledge Domain Action Plan`
  - `Benchmark Artifact Bundle Knowledge Domain Action Plan`
  - `Benchmark Companion Knowledge Domain Action Plan`
  - `Benchmark Release Decision Knowledge Domain Action Plan`
  - `Benchmark Release Runbook Knowledge Domain Action Plan`
- Signal lights now include the standalone action-plan readiness state:
  - ready -> green
  - blocked -> red
  - partial -> yellow
