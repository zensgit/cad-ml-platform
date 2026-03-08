# Benchmark Companion Summary CI Validation

## Goal

Wire the standalone benchmark companion summary into `evaluation-report.yml` so CI can
build, upload, and surface a compact benchmark operator summary.

## Changes

- Added workflow-dispatch inputs for benchmark companion summary overrides.
- Added environment variables for companion summary title, inputs, and outputs.
- Added `Build benchmark companion summary (optional)` step.
- Added `Upload benchmark companion summary` artifact step.
- Added job-summary lines for:
  - overall status
  - review surface
  - primary gap
  - hybrid / assistant / review-queue / OCR / Qdrant statuses
  - blockers
  - recommended actions
  - artifact path

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

## Expected Outcome

- CI can optionally build a benchmark companion summary from scorecard,
  operational summary, and artifact bundle inputs.
- The summary appears in uploaded artifacts and GitHub Actions job summary.
- Workflow regression coverage remains green.
