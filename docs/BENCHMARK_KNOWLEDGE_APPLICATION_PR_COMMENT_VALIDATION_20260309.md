# Benchmark Knowledge Application PR Comment Validation

## Scope

This delivery extends the PR comment and signal-light surface in
`.github/workflows/evaluation-report.yml` so benchmark knowledge application
signals appear alongside benchmark knowledge readiness, knowledge drift,
engineering, and real-data signals.

## Changes

- Added PR comment variables for:
  - benchmark knowledge application exporter status
  - ready / partial / missing / total domain counts
  - focus areas, priority domains, domain statuses, and recommendations
- Extended downstream PR comment rows with:
  - artifact bundle knowledge application
  - companion summary knowledge application
  - release decision knowledge application
  - release runbook knowledge application
- Added a dedicated signal-light row:
  - `Benchmark Knowledge Application`
- Extended workflow regression tests to assert the new PR comment variables,
  rows, and recommendation fragments.

## Validation

Commands:

```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
python3 - <<'PY'
from pathlib import Path
import yaml
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text(encoding='utf-8'))
print('yaml-ok')
PY
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Results:

- `py_compile`: passed
- `flake8`: passed
- workflow YAML parse: passed
- `pytest`: passed

## Notes

- This layer only wires PR comment and signal-light visibility.
- Exporter logic, downstream surface plumbing, and workflow job-summary wiring
  are covered by the earlier stacked knowledge-application deliveries.
