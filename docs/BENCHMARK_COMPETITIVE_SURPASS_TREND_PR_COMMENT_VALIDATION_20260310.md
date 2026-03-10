# Benchmark Competitive Surpass Trend PR Comment Validation

## Scope
- Added `competitive_surpass_trend` coverage to GitHub PR comments and signal lights.
- Extended standalone, artifact bundle, companion summary, release decision, and
  release runbook PR comment rows.

## Key Changes
- Added PR comment constants for:
  - `benchmarkCompetitiveSurpassTrend*`
  - `benchmarkArtifactBundleCompetitiveSurpassTrend*`
  - `benchmarkCompanionCompetitiveSurpassTrend*`
  - `benchmarkReleaseCompetitiveSurpassTrend*`
  - `benchmarkReleaseRunbookCompetitiveSurpassTrend*`
- Added standalone and downstream status lines.
- Added standalone and downstream signal lights.
- Added PR comment table rows for all five trend surfaces.
- Extended workflow contract assertions in
  `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`.

## Validation
```bash
python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path('.github/workflows/evaluation-report.yml').read_text())
print('yaml_ok')
PY

python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Result
- YAML parse: passed
- `py_compile`: passed
- `flake8`: passed
- `pytest`: `3 passed, 1 warning`

## Outcome
- PR comments now expose `competitive_surpass_trend` consistently across the
  benchmark control plane.
- Signal lights now distinguish trend states:
  - `improved` / `stable` -> green
  - `regressed` -> red
  - `mixed` / `baseline_missing` -> yellow
