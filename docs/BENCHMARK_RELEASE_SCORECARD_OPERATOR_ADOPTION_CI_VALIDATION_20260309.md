# Benchmark Release Scorecard Operator Adoption CI Validation

## Scope

- Wire release decision scorecard/operational operator-adoption outputs into
  `evaluation-report.yml`.
- Wire release runbook scorecard/operational operator-adoption outputs into
  `evaluation-report.yml`.
- Expose both surfaces in GitHub job summary.

## Design

- `Build benchmark release decision (optional)` now exports:
  - `scorecard_operator_adoption_status`
  - `scorecard_operator_adoption_mode`
  - `scorecard_operator_adoption_knowledge_outcome_drift_status`
  - `scorecard_operator_adoption_knowledge_outcome_drift_summary`
  - `operational_operator_adoption_status`
  - `operational_operator_adoption_knowledge_outcome_drift_status`
  - `operational_operator_adoption_knowledge_outcome_drift_summary`
- `Build benchmark release runbook (optional)` now:
  - receives `--benchmark-scorecard`
  - receives `--benchmark-operational-summary`
  - exports the same scorecard/operational operator-adoption fields
- Job summary prints both release decision and release runbook scorecard /
  operational operator-adoption lines.

## Files

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation

```bash
python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path('.github/workflows/evaluation-report.yml').read_text())
print('yaml-ok')
PY

python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Result

- YAML parse: pass
- `py_compile`: pass
- `flake8`: pass
- `pytest`: pass
