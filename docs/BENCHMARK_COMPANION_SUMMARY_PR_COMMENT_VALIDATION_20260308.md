# Benchmark Companion Summary PR Comment Validation

## Goal

Expose benchmark companion summary status directly in the PR comment and signal-light
table so reviewers can see the compact benchmark state without opening artifacts.

## Changes

- Added PR-comment variables for benchmark companion summary outputs:
  - overall status
  - review surface
  - primary gap
  - hybrid / assistant / review-queue / OCR / Qdrant statuses
  - blockers
  - recommended actions
  - artifact path
- Added a `Benchmark Companion Summary` row to the main PR comment table.
- Added a `Benchmark Companion Summary` row to the signal-light table.

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

- PR comments show benchmark companion summary status alongside benchmark scorecard,
  operational summary, assistant evidence, review queue, and OCR review signals.
- Reviewers can use the compact benchmark row as the default operator-facing entry point.
