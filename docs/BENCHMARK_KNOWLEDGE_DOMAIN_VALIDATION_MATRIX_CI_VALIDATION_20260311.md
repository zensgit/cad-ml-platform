# Benchmark Knowledge Domain Validation Matrix CI Validation

## Goal
- Wire `benchmark_knowledge_domain_validation_matrix` into the benchmark control-plane
  workflow so it becomes a first-class CI artifact instead of only a local exporter.
- Feed the same signal into bundle, companion, release decision, and release runbook
  builders through `evaluation-report.yml`.

## Scope
- `workflow_dispatch` inputs
- environment defaults
- optional build step
- artifact upload
- job summary
- downstream `bundle / companion / release decision / release runbook` passthrough
- workflow contract tests

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
- `pytest`: `6 passed`

## Outcome
- CI can now build and upload `benchmark_knowledge_domain_validation_matrix`.
- Bundle, companion, release decision, and release runbook builders now accept the
  validation matrix from workflow inputs or upstream step outputs.
- Job summary now surfaces validation-matrix status and downstream status propagation.
