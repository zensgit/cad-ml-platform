# Benchmark Knowledge Domain Release Readiness Matrix PR Comment Validation

## Scope
- Added PR comment and signal-light wiring for `benchmark_knowledge_domain_release_readiness_matrix`.
- Extended downstream PR comment surfaces for:
  - artifact bundle
  - companion summary
  - release decision
  - release runbook
- Repaired the existing `knowledge_domain_release_gate` PR-comment script join so the
  script remains syntactically coherent after adding the new readiness-matrix lines.

## Files
- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

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
git diff --check
```

## Expected Coverage
- PR comment exposes standalone readiness-matrix status line.
- PR comment exposes bundle / companion / release decision / release runbook
  readiness-matrix status lines.
- Signal lights expose standalone, release decision, and release runbook
  readiness-matrix states.
- Workflow continues to expose existing knowledge reference inventory PR-comment
  lines without regression.

## Result
- `python3 -m py_compile ...` passed
- `flake8 ... --max-line-length=100` passed
- workflow YAML parse passed (`yaml-ok`)
- `pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`:
  `7 passed, 1 warning`
- `git diff --check` passed
