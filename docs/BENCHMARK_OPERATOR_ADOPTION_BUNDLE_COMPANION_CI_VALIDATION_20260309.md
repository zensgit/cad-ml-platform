# Benchmark Operator Adoption Bundle / Companion CI Validation

## Scope
- Export bundle / companion operator adoption scorecard and operational fields to workflow outputs
- Show these fields in job summary
- Show these fields in PR comment detail rows

## Added Workflow Outputs
- `scorecard_operator_adoption_status`
- `scorecard_operator_adoption_mode`
- `scorecard_operator_adoption_knowledge_outcome_drift_status`
- `scorecard_operator_adoption_knowledge_outcome_drift_summary`
- `operational_operator_adoption_status`
- `operational_operator_adoption_knowledge_outcome_drift_status`
- `operational_operator_adoption_knowledge_outcome_drift_summary`

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
