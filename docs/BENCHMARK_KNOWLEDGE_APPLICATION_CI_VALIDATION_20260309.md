# Benchmark Knowledge Application CI Validation

## Scope
- Add `benchmark_knowledge_application` inputs and env wiring to `evaluation-report.yml`
- Build and upload `benchmark_knowledge_application` artifact in CI
- Pass knowledge-application artifacts into:
  - benchmark artifact bundle
  - benchmark companion summary
  - benchmark release decision
  - benchmark release runbook
- Surface knowledge-application status in GitHub job summary

## Key Files
- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Added CI Surfaces
- `Build benchmark knowledge application (optional)`
- `Upload benchmark knowledge application`
- Job summary rows for:
  - benchmark knowledge application
  - bundle knowledge application
  - companion knowledge application
  - release knowledge application
  - release runbook knowledge application

## Validation
```bash
python3 - <<'PY'
from pathlib import Path
import yaml
yaml.safe_load(Path('.github/workflows/evaluation-report.yml').read_text())
print('yaml-ok')
PY

python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Result
- YAML parse: passed
- `py_compile`: passed
- `pytest`: `3 passed`

## Notes
- This branch only wires CI/artifact/job-summary surfaces.
- PR comment / signal lights are intentionally stacked in the next branch to avoid large single-branch workflow diffs.
