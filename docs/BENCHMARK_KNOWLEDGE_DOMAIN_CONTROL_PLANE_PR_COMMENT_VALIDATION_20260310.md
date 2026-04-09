# Benchmark Knowledge Domain Control Plane PR Comment Validation

## Scope
- Add `knowledge_domain_control_plane` PR comment rows and signal light wiring.
- Extend `evaluation-report.yml` to surface control-plane status in:
  - main benchmark PR comment table
  - artifact bundle row
  - companion summary row
  - release decision row
  - release runbook row
  - signal lights section

## Files
- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Added PR Comment Outputs
- `Benchmark Knowledge Domain Control Plane`
- `Benchmark Artifact Bundle Knowledge Domain Control Plane`
- `Benchmark Companion Knowledge Domain Control Plane`
- `Benchmark Release Decision Knowledge Domain Control Plane`
- `Benchmark Release Runbook Knowledge Domain Control Plane`
- signal light: `Benchmark Knowledge Domain Control Plane`

## Added Status Fields
- `status`
- `ready`
- `partial`
- `blocked`
- `missing`
- `total`
- `actions`
- `high`
- `release_blockers`
- `priority_domains`
- `focus_areas`
- `recommendations`
- `artifact`

## Validation
```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path('.github/workflows/evaluation-report.yml').read_text())
print('yaml_ok')
PY
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Result
- `py_compile`: passed
- `flake8`: passed
- workflow YAML parse: passed
- `pytest`: `3 passed, 1 warning`

## Notes
- The implementation mirrors the existing `knowledge_domain_action_plan` PR comment pattern.
- No edits were made in the main workspace; all work stayed in an isolated worktree.
