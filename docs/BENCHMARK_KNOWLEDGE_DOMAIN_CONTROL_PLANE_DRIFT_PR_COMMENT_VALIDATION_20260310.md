# Benchmark Knowledge Domain Control Plane Drift PR Comment Validation

## Scope
- Add PR comment and signal-light coverage for `benchmark_knowledge_domain_control_plane_drift`.
- Surface direct drift status plus downstream bundle / companion / release decision /
  release runbook drift rows in `evaluation-report.yml`.

## Files
- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## What Changed
- Added job-summary lines for standalone `knowledge_domain_control_plane_drift`.
- Added PR comment status rows for:
  - `Benchmark Knowledge Domain Control Plane Drift`
  - `Benchmark Artifact Bundle Knowledge Domain Control Plane Drift`
  - `Benchmark Companion Knowledge Domain Control Plane Drift`
  - `Benchmark Release Decision Knowledge Domain Control Plane Drift`
  - `Benchmark Release Runbook Knowledge Domain Control Plane Drift`
- Added direct signal light for `Benchmark Knowledge Domain Control Plane Drift`.
- Added workflow-contract assertions for the new summary/comment wiring.

## Validation
```bash
python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path('.github/workflows/evaluation-report.yml').read_text())
print('yaml_ok')
PY

python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

git diff --check
```

## Result
- YAML parse: passed
- `py_compile`: passed
- `flake8`: passed
- `pytest`: `3 passed`
- `git diff --check`: passed
