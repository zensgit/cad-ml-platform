# Benchmark Knowledge Domain Control Plane Drift CI Validation

## Scope
- Wire `benchmark_knowledge_domain_control_plane_drift` into `evaluation-report.yml`
- Propagate drift payloads into:
  - artifact bundle
  - companion summary
  - release decision
  - release runbook
- Add workflow contract coverage for:
  - dispatch inputs
  - build step
  - upload step
  - downstream exporter passthrough

## Key Changes
- Added workflow dispatch inputs:
  - `benchmark_knowledge_domain_control_plane_drift_enable`
  - `benchmark_knowledge_domain_control_plane_drift_current_summary_json`
  - `benchmark_knowledge_domain_control_plane_drift_previous_summary_json`
- Added workflow env defaults for control-plane drift output paths and downstream JSON passthrough.
- Added build step:
  - `Build benchmark knowledge domain control plane drift (optional)`
- Added upload step:
  - `Upload benchmark knowledge domain control plane drift`
- Extended downstream exporter invocations with:
  - `--benchmark-knowledge-domain-control-plane-drift`
- Extended downstream `GITHUB_OUTPUT` extraction with:
  - `knowledge_domain_control_plane_drift_status`
  - `knowledge_domain_control_plane_drift_domain_regressions`
  - `knowledge_domain_control_plane_drift_domain_improvements`
  - `knowledge_domain_control_plane_drift_resolved_release_blockers`
  - `knowledge_domain_control_plane_drift_new_release_blockers`
  - `knowledge_domain_control_plane_drift_recommendations`

## Validation
Commands run:

```bash
python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path('.github/workflows/evaluation-report.yml').read_text())
print('yaml_ok')
PY

python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

git diff --check
```

Results:
- YAML parse: passed
- `py_compile`: passed
- `flake8`: passed
- `pytest`: `3 passed, 1 warning`
- `git diff --check`: passed

## Notes
- This branch only wires CI / downstream flow.
- Job summary and PR comment rendering for control-plane drift should be stacked in the next branch to keep the diff narrow and reduce conflicts in `evaluation-report.yml`.
