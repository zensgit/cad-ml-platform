# Benchmark Operator Adoption Release Alignment CI Validation

## Scope

Wire the new `release_surface_alignment` fields from
`scripts/export_benchmark_operator_adoption.py` into the benchmark operator
adoption workflow outputs and job summary.

## Files

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Added Workflow Outputs

- `release_surface_alignment_status`
- `release_surface_alignment_summary`
- `release_surface_alignment_mismatches`

## Added Job Summary Lines

- `Benchmark operator adoption release surface alignment`
- `Benchmark operator adoption release surface alignment summary`
- `Benchmark operator adoption release surface mismatches`

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
