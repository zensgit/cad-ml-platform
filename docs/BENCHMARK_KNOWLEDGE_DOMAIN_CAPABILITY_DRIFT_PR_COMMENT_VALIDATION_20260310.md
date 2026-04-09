# Benchmark Knowledge Domain Capability Drift PR Comment Validation

## Scope

- Added `knowledge_domain_capability_drift` signal-light and status lines to the PR comment surface.
- Extended downstream PR comment rows for:
  - `benchmark artifact bundle`
  - `benchmark companion summary`
  - `benchmark release decision`
  - `benchmark release runbook`

## Validation

```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

python3 - <<'PY'
import yaml, pathlib
path = pathlib.Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text())
print('yaml-ok')
PY

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Result

- `py_compile` passed
- `flake8` passed
- workflow YAML parse passed
- `pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py` passed: `3 passed, 1 warning`
