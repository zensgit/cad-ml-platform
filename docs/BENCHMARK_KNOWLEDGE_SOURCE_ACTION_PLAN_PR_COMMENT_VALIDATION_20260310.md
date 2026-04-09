# Benchmark Knowledge Source Action Plan PR Comment Validation

## Goal

Expose `benchmark_knowledge_source_action_plan` through PR comment rows and
signal lights so benchmark reviewers can see knowledge-source expansion status
without opening artifacts.

## Scope

- Add PR comment state for the standalone `knowledge_source_action_plan`.
- Add downstream PR comment rows for:
  - artifact bundle
  - companion summary
  - release decision
  - release runbook
- Add a `Benchmark Knowledge Source Action Plan` signal light.
- Extend workflow regression coverage in
  `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`.

## Key Files

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation

```bash
python3 - <<'PY'
from pathlib import Path
import yaml
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text(encoding='utf-8'))
print('yaml_ok')
PY

python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Result

- Workflow YAML parse: passed
- `py_compile`: passed
- `flake8`: passed
- `pytest`: `3 passed`

## Notes

- This branch only covers PR comment and signal-light exposure.
- CI build/upload/job-summary wiring stays on the stacked CI branch.
