# Benchmark Artifact Bundle CI Validation

## Goal

Wire the benchmark artifact bundle export into `evaluation-report.yml` so CI can
optionally build, upload, and summarize a single benchmark artifact bundle.

## Changes

- Added `workflow_dispatch` inputs for benchmark artifact bundle sources.
- Added environment variables for benchmark artifact bundle enablement, input
  paths, and output paths.
- Added optional step:
  - `Build benchmark artifact bundle (optional)`
- Added artifact upload step:
  - `Upload benchmark artifact bundle`
- Added job summary lines covering:
  - overall status
  - available artifact count
  - feedback status
  - assistant status
  - review queue status
  - OCR status
  - blockers
  - recommendations
  - artifact markdown path
- Extended workflow regression tests to cover the new inputs, env vars, build
  step, upload step, and summary lines.

## Validation

Commands run:

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

Results:

- `py_compile`: passed
- `flake8`: passed
- `pytest`: `3 passed`
- YAML parse: `yaml_ok`

## Notes

- This change only wires the bundle into CI and job summary output.
- PR comment surfacing is intentionally handled in a follow-up stacked branch.
