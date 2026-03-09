# Benchmark Real-Data Scorecard PR Comment Validation

## Goal
- Expose `benchmark_realdata_scorecard` in the PR comment and signal-light section of `evaluation-report.yml`.
- Surface the scorecard across downstream benchmark release surfaces:
  - artifact bundle
  - companion summary
  - release decision
  - release runbook

## Key Changes
- Added PR-comment bindings for:
  - `benchmarkRealdataScorecard*`
  - `benchmarkArtifactBundleRealdataScorecard*`
  - `benchmarkCompanionRealdataScorecard*`
  - `benchmarkReleaseRealdataScorecard*`
  - `benchmarkReleaseRunbookRealdataScorecard*`
- Added top-level status line:
  - `Benchmark Real-Data Scorecard`
- Added downstream rows:
  - `Benchmark Artifact Bundle Real-Data Scorecard`
  - `Benchmark Companion Real-Data Scorecard`
  - `Benchmark Release Decision Real-Data Scorecard`
  - `Benchmark Release Runbook Real-Data Scorecard`
- Added top-level signal light:
  - `Benchmark Real-Data Scorecard`
- Extended workflow regression assertions to cover the new comment/status bindings.

## Files
- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation
```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
python3 - <<'PY'
import yaml
from pathlib import Path
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text(encoding='utf-8'))
print('yaml_ok')
PY
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Result
- `py_compile`: passed
- workflow YAML parse: passed
- `flake8`: passed
- `pytest`: `3 passed, 1 warning`
