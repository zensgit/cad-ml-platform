# Benchmark Knowledge Domain Release Readiness Matrix CI Validation

## Objective
- Wire `knowledge_domain_release_readiness_matrix` into `evaluation-report.yml`.
- Expose the component in CI artifact upload, job summary, and downstream benchmark surfaces.
- Lock the workflow contract with unit tests.

## Workflow Surfaces
- Standalone build step:
  - `Build benchmark knowledge domain release readiness matrix (optional)`
- Artifact upload:
  - `Upload benchmark knowledge domain release readiness matrix`
- Downstream passthrough:
  - `benchmark artifact bundle`
  - `benchmark companion summary`
  - `benchmark release decision`
  - `benchmark release runbook`
- Job summary rows:
  - standalone readiness matrix
  - bundle readiness matrix
  - companion readiness matrix
  - release decision readiness matrix
  - release runbook readiness matrix

## Validation Commands
```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
python3 - <<'PY'
from pathlib import Path
import yaml
yaml.safe_load(Path('.github/workflows/evaluation-report.yml').read_text())
print('yaml-ok')
PY
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Result
- `python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py` passed
- `flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100` passed
- workflow YAML parse passed
- `pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py` passed: `6 passed, 1 warning`
- `git diff --check` passed
