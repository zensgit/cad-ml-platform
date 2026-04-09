# Benchmark Knowledge Source Coverage PR Comment Validation

## Scope
- Added `knowledge_source_coverage` PR comment and signal-light wiring to
  `.github/workflows/evaluation-report.yml`.
- Extended workflow contract tests to assert the new coverage constants,
  downstream status lines, markdown rows, and signal-light entries.

## Files
- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

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
```

## Result
- Workflow YAML parses successfully.
- Python compile passes.
- Flake8 passes.
- Workflow contract pytest passes.
