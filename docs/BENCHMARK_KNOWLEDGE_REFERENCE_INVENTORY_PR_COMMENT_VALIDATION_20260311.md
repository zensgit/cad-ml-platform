# Benchmark Knowledge Reference Inventory PR Comment Validation

## Goal

Expose `benchmark_knowledge_reference_inventory` in:

- PR comment rows
- signal lights
- bundle / companion downstream status rows

## Scope

- Add workflow JavaScript bindings for standalone knowledge reference inventory
- Add bundle / companion summary bindings
- Add status lines and signal light
- Add PR comment coverage test

## Validation

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

Expected:

- workflow YAML parses
- workflow contract tests pass
- PR comment contains standalone / bundle / companion knowledge reference inventory rows
- signal lights include `Benchmark Knowledge Reference Inventory`
