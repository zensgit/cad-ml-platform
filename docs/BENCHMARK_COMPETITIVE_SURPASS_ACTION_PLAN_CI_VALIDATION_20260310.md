# Benchmark Competitive Surpass Action Plan CI Validation

## Scope
- Wire `benchmark_competitive_surpass_action_plan` into `evaluation-report.yml`.
- Pass action-plan payloads through benchmark downstream surfaces:
  - artifact bundle
  - companion summary
  - release decision
  - release runbook
- Expose action-plan status in job summary and artifact uploads.

## Workflow Changes
- Added `workflow_dispatch` inputs for standalone action-plan export.
- Added downstream JSON override inputs for:
  - `benchmark_artifact_bundle_competitive_surpass_action_plan_json`
  - `benchmark_companion_summary_competitive_surpass_action_plan_json`
  - `benchmark_release_decision_competitive_surpass_action_plan_json`
  - `benchmark_release_runbook_competitive_surpass_action_plan_json`
- Added environment defaults for standalone and downstream action-plan JSON paths.
- Added `Build benchmark competitive surpass action plan (optional)` step.
- Added `Upload benchmark competitive surpass action plan` artifact step.
- Added downstream output extraction for:
  - `competitive_surpass_action_plan_status`
  - `competitive_surpass_action_plan_total_action_count`
  - `competitive_surpass_action_plan_priority_pillars`
  - `competitive_surpass_action_plan_recommendations`
- Added job summary lines for standalone and downstream action-plan status.

## Test Coverage
- Updated workflow contract coverage in:
  - `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
- Assertions cover:
  - dispatch inputs
  - standalone build step
  - upload step
  - artifact bundle / companion / release / runbook CLI flags
  - downstream output writes

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
- This branch only handles CI wiring and job summary exposure.
- PR comment / signal-light exposure should land as the next stacked branch.
