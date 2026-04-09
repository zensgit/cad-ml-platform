# Benchmark Operator Adoption Knowledge Outcome Drift CI Validation

## Goal
- Wire `benchmark_operator_adoption` consumption of `knowledge_outcome_drift` into CI, job summary, and PR comment surfaces.
- Keep the workflow contract aligned with the exporter added in `feat/benchmark-operator-adoption-knowledge-outcome-drift`.

## Scope
- Add workflow dispatch input and env passthrough for `benchmark_operator_adoption_knowledge_outcome_drift_json`.
- Update both duplicated `Build benchmark operator adoption (optional)` steps.
- Export `knowledge_outcome_drift_status` and `knowledge_outcome_drift_summary` to step outputs.
- Extend job summary and PR comment status lines with operator-adoption outcome-drift detail.
- Lock the behavior in `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`.

## Files
- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation
```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path('.github/workflows/evaluation-report.yml').read_text())
print('yaml_ok')
PY
git diff --check
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Result
- `py_compile`: passed
- `flake8`: passed
- workflow YAML parse: passed
- `git diff --check`: passed
- `pytest`: `3 passed, 1 warning`

## Outcome
- Operator-adoption benchmark now exposes both knowledge drift and knowledge outcome drift on CI surfaces.
- Job summary and PR comment can distinguish:
  - `knowledge_drift`
  - `knowledge_outcome_drift`
- The workflow contract test now guards the new input, flags, outputs, and summary lines.
