# Benchmark Knowledge Domain Surface Action Plan CI Validation

## Scope
- Wired `benchmark_knowledge_domain_surface_action_plan` into `evaluation-report.yml`.
- Added:
  - workflow dispatch inputs
  - environment defaults
  - optional build step
  - artifact upload
  - job summary output
  - downstream passthrough to bundle, companion, release decision, and release runbook

## Validation
Executed in `/private/tmp/cad-ml-platform-knowledge-domain-surface-action-plan-ci-20260312`.

```bash
python3 - <<'PY'
from pathlib import Path
import yaml
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text(encoding='utf-8'))
print('yaml-ok')
PY
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py -q
```

## Results
- workflow YAML parse: passed
- `py_compile`: passed
- `flake8`: passed
- `pytest`: `12 passed`

## Notes
- This branch intentionally limits scope to CI build/upload/summary/passthrough.
- PR comment and signal-light exposure are handled in the stacked follow-up branch.
