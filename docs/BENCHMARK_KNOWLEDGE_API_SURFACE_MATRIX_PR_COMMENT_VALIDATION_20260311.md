# Benchmark Knowledge API Surface Matrix PR Comment Validation

## Goal

Expose the benchmark knowledge API surface matrix in PR comments and signal
lights so missing public API coverage is visible during review, not only in
artifacts.

## Scope

- PR comment status line
- PR signal light
- top-level benchmark table row

## Verification

Commands:

```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Additional parse check:

```bash
python3 - <<'PY'
from pathlib import Path
import yaml
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text(encoding='utf-8'))
print('yaml-ok')
PY
```

## Expected Result

The PR comment now includes a `Benchmark Knowledge Domain API Surface Matrix`
row with:

- overall status
- domain counts
- total public API route count
- public API gap domains
- reference gap domains
- recommendations
