# Benchmark Release Runbook CI Validation 2026-03-08

## Goal

Wire the standalone benchmark release runbook exporter into `evaluation-report.yml`
so CI can build, upload, and summarize operator-facing release actions.

## Delivered

- workflow dispatch inputs for release runbook JSON sources
- workflow env for runbook title and output paths
- optional `Build benchmark release runbook` step
- benchmark release runbook artifact upload
- job summary lines for:
  - release status
  - freeze readiness
  - next action
  - missing artifacts
  - blocking signals
  - review signals

## Validation

```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
python3 - <<'PY'
from pathlib import Path
import yaml
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text(encoding='utf-8'))
print('yaml_ok')
PY
```
