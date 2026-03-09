# Benchmark Operator Adoption Release Alignment PR Comment Validation

## Scope

Expose the new operator-adoption `release_surface_alignment` fields in the PR
comment and signal-light block emitted by `evaluation-report.yml`.

## Files

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Added PR Comment Rows

- `Benchmark Operator Adoption Release Surface Alignment`
- `Benchmark Operator Adoption Release Surface Mismatches`

## Added Signal Light

- `Benchmark Operator Adoption Release Surface Alignment`

## Validation

Executed:

```bash
python3 - <<'PY'
from pathlib import Path
import yaml
yaml.safe_load(Path(".github/workflows/evaluation-report.yml").read_text())
print("yaml-ok")
PY

python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```
