# Benchmark Operator Adoption CI Validation

## Scope
- Add optional benchmark operator adoption export to `evaluation-report.yml`
- Upload operator adoption artifact when enabled
- Surface operator adoption readiness, mode, and actions in job summary

## Files
- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation
```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
python3 - <<'PY'
from pathlib import Path
import yaml
workflow = Path(".github/workflows/evaluation-report.yml")
yaml.safe_load(workflow.read_text(encoding="utf-8"))
print("yaml_ok")
PY
```

## Result
- Workflow dispatch inputs include operator adoption overrides
- Workflow env includes operator adoption input/output paths
- Build, upload, and job-summary steps are covered by regression tests
- Operator adoption exporter is wired to release decision, runbook, review queue, and feedback flywheel inputs
