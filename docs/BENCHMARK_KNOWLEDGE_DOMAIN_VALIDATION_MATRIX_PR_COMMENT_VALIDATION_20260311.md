# Benchmark Knowledge Domain Validation Matrix PR Comment Validation

## Goal
- Surface `benchmark_knowledge_domain_validation_matrix` in the GitHub PR comment and
  signal-light section so reviewers can see the knowledge validation gap without opening
  artifacts.

## Scope
- PR comment status line
- signal light
- benchmark results table row

## Files
- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Verification
```bash
python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path('.github/workflows/evaluation-report.yml').read_text())
print('yaml-ok')
PY

python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Result
- workflow YAML parse: pass
- `py_compile`: pass
- `flake8`: pass
- `pytest`: `7 passed`

## Outcome
- PR comments now include `Benchmark Knowledge Domain Validation Matrix`.
- Signal lights now reflect `ready / blocked / partial` semantics for this new benchmark
  component.
