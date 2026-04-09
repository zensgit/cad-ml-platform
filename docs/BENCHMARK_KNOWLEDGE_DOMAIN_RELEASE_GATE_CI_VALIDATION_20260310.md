# Benchmark Knowledge Domain Release Gate CI Validation

## Scope
- Wire `benchmark_knowledge_domain_release_gate` into `evaluation-report.yml`
- Pass release-gate JSON into:
  - `benchmark_artifact_bundle`
  - `benchmark_companion_summary`
- Expose build/upload/job-summary signals for downstream CI and later PR comment layers

## Key Changes
- Added `workflow_dispatch` inputs for:
  - `benchmark_knowledge_domain_release_gate_enable`
  - capability matrix / capability drift / action plan / control plane / control-plane drift / release-surface alignment JSON inputs
- Added workflow env defaults for:
  - release-gate input JSON paths
  - output JSON / output MD paths
  - artifact-bundle / companion-summary passthrough JSON vars
- Added workflow step:
  - `Build benchmark knowledge domain release gate (optional)`
- Added artifact upload step:
  - `Upload benchmark knowledge domain release gate`
- Extended bundle / companion build steps to accept:
  - `--benchmark-knowledge-domain-release-gate`
- Extended bundle / companion outputs with:
  - `knowledge_domain_release_gate_status`
  - `knowledge_domain_release_gate_summary`
  - `knowledge_domain_release_gate_gate_open`
  - `knowledge_domain_release_gate_blocking_reasons`
  - `knowledge_domain_release_gate_releasable_domains`
  - `knowledge_domain_release_gate_blocked_domains`
  - `knowledge_domain_release_gate_priority_domains`
  - `knowledge_domain_release_gate_recommendations`
- Added job summary lines for:
  - standalone release-gate status
  - artifact-bundle release-gate status
  - companion release-gate status

## Files
- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation
```bash
python3 - <<'PY'
import yaml
from pathlib import Path
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text())
print('yaml-ok')
PY

python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

git diff --check
```

## Result
- YAML parse: passed
- `py_compile`: passed
- `flake8`: passed
- `pytest`: `3 passed, 1 warning`
- `git diff --check`: passed

## Notes
- This layer only wires CI/build/upload/summary/passthrough.
- PR comment / signal-light exposure is intentionally stacked into the next branch.
