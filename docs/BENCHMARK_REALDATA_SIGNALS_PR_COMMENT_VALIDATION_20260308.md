# Benchmark Real-Data Signals PR Comment Validation

## Scope

This delivery extends the PR comment / signal-light surface in
`.github/workflows/evaluation-report.yml` so benchmark real-data signals appear
alongside scorecard, engineering, knowledge, and operator-adoption signals.

## Changes

- Added PR comment variables for:
  - benchmark real-data exporter status
  - ready / partial / blocked / available component counts
  - hybrid / history / STEP smoke / STEP dir sub-status
  - real-data recommendations and artifact path
- Extended downstream PR comment rows with:
  - artifact bundle real-data status
  - companion summary real-data status
  - release decision real-data status
  - release runbook real-data status
- Added a dedicated signal-light row:
  - `Benchmark Real-Data Signals`

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

- This layer is PR-comment only.
- Exporter logic and workflow job-summary wiring remain in the previous stacked
  delivery.
