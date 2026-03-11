# Benchmark Knowledge Domain Release Readiness Drift CI Validation

## Scope
- Wire `benchmark_knowledge_domain_release_readiness_drift` into `evaluation-report.yml`
- Propagate drift artifact inputs into:
  - benchmark artifact bundle
  - benchmark companion summary
  - benchmark release decision
  - benchmark release runbook
- Expose standalone and downstream drift status in job summary

## Key Changes
- Added workflow dispatch inputs for release-readiness drift JSON overrides
- Added workflow env defaults for current/previous summaries and downstream JSONs
- Added optional build step:
  - `Build benchmark knowledge domain release readiness drift (optional)`
- Added artifact upload step:
  - `Upload benchmark knowledge domain release readiness drift`
- Added downstream `add_if_exists` wiring for bundle, companion, release decision, and release runbook
- Added downstream output exports:
  - `knowledge_domain_release_readiness_drift_status`
  - `knowledge_domain_release_readiness_drift_summary`
  - `knowledge_domain_release_readiness_drift_domain_regressions`
  - `knowledge_domain_release_readiness_drift_domain_improvements`
  - `knowledge_domain_release_readiness_drift_recommendations`
- Added standalone and downstream summary lines to `Create job summary`

## Validation
```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
python3 - <<'PY'
from pathlib import Path
import yaml
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text(encoding='utf-8'))
print('yaml_ok')
PY
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
git diff --check
```

## Results
- `py_compile`: passed
- `flake8`: passed
- workflow YAML parse: `yaml_ok`
- `pytest`: `8 passed, 1 warning`
- `git diff --check`: passed

## Notes
- Warning is the existing `PendingDeprecationWarning` from `starlette` multipart parsing, not introduced by this change.
