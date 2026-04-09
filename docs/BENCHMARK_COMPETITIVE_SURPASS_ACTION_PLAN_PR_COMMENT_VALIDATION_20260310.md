# Benchmark Competitive Surpass Action Plan PR Comment Validation

## Scope
- Expose `benchmark_competitive_surpass_action_plan` in PR comment and signal lights.
- Surface action-plan status for:
  - standalone benchmark block
  - artifact bundle
  - companion summary
  - release decision
  - release runbook

## Workflow Changes
- Added PR comment constants for action-plan status, counts, pillars, recommendations, and artifact path.
- Added status-line formatting for:
  - `Benchmark Competitive Surpass Action Plan`
  - `Benchmark Artifact Bundle Competitive Surpass Action Plan`
  - `Benchmark Companion Competitive Surpass Action Plan`
  - `Benchmark Release Decision Competitive Surpass Action Plan`
  - `Benchmark Release Runbook Competitive Surpass Action Plan`
- Added signal-light evaluation for the same five surfaces.
- Added markdown rows to:
  - benchmark summary table
  - graph2d signal lights table

## Test Coverage
- Updated:
  - `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
- Assertions cover:
  - action-plan JS constants
  - action-plan status lines
  - action-plan signal lights
  - rendered markdown labels

## Validation
```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
python3 - <<'PY'
import yaml, pathlib
yaml.safe_load(pathlib.Path('.github/workflows/evaluation-report.yml').read_text())
print('yaml-ok')
PY
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Result
- `py_compile`: pass
- `flake8`: pass
- workflow YAML parse: pass
- `pytest`: `3 passed`

## Notes
- This branch only changes PR comment / signal-light presentation.
- CI build/upload/downstream output wiring remains in the parent stacked branch.
