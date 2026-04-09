# Benchmark Operator Adoption Scorecard / Operational CI Validation

## Scope
- Wire scorecard `operator_adoption` fields into `evaluation-report.yml`
- Wire operational summary `operator_adoption` fields into `evaluation-report.yml`
- Expose both surfaces in job summary and PR comment / signal lights

## Changes
- Added scorecard workflow inputs/env passthrough for operator adoption summary JSON
- Added operational workflow inputs/env passthrough for operator adoption JSON
- Exported new scorecard outputs:
  - `operator_adoption_status`
  - `operator_adoption_mode`
  - `operator_adoption_knowledge_outcome_drift_status`
  - `operator_adoption_knowledge_outcome_drift_summary`
- Exported new operational outputs:
  - `operator_adoption_status`
  - `operator_adoption_knowledge_outcome_drift_status`
  - `operator_adoption_knowledge_outcome_drift_summary`
- Added scorecard / operational operator adoption lines to:
  - job summary
  - PR comment detail table
  - PR signal lights

## Validation
```bash
python3 - <<'PY'
import yaml
from pathlib import Path
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text(encoding='utf-8'))
print('yaml-ok')
PY

python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Expected Result
- Workflow parses cleanly
- Regression test asserts scorecard / operational operator adoption fields in:
  - build steps
  - job summary
  - PR comment script
