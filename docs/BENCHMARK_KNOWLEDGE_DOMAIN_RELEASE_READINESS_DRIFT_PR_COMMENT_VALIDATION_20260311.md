# Benchmark Knowledge Domain Release Readiness Drift PR Comment Validation

## Scope
- Extend PR comment and signal lights for `benchmark_knowledge_domain_release_readiness_drift`
- Surface drift status across:
  - standalone benchmark line
  - artifact bundle
  - companion summary
  - release decision
  - release runbook

## Key Changes
- Added PR-comment script variables:
  - `benchmarkKnowledgeDomainReleaseReadinessDriftStatusLine`
  - `benchmarkArtifactBundleKnowledgeDomainReleaseReadinessDriftStatusLine`
  - `benchmarkCompanionKnowledgeDomainReleaseReadinessDriftStatusLine`
  - `benchmarkReleaseKnowledgeDomainReleaseReadinessDriftStatusLine`
  - `benchmarkReleaseRunbookKnowledgeDomainReleaseReadinessDriftStatusLine`
- Added signal-light variables:
  - `benchmarkKnowledgeDomainReleaseReadinessDriftLight`
  - `benchmarkReleaseDecisionKnowledgeDomainReleaseReadinessDriftLight`
  - `benchmarkReleaseRunbookKnowledgeDomainReleaseReadinessDriftLight`
- Added PR comment rows for standalone and downstream release-readiness drift
- Added workflow contract coverage for the new PR comment lines

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
- `pytest`: `9 passed, 1 warning`
- `git diff --check`: passed

## Notes
- Warning is the existing `PendingDeprecationWarning` from `starlette` multipart parsing, not introduced by this change.
