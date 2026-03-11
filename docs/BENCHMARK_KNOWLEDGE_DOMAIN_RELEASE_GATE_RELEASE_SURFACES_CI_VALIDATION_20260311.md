# Benchmark Knowledge Domain Release Gate Release Surfaces CI Validation

## Scope
- Wire `knowledge_domain_release_gate` into the release decision and release
  runbook benchmark surfaces consumed by `evaluation-report.yml`.
- Expose the new release-gate fields in workflow inputs, summary rows, and CI
  contract assertions.

## Key Changes
- Added `workflow_dispatch` inputs:
  - `benchmark_release_decision_knowledge_domain_release_gate_json`
  - `benchmark_release_runbook_knowledge_domain_release_gate_json`
- Passed `--benchmark-knowledge-domain-release-gate` through both release
  exporters.
- Emitted release decision / runbook workflow outputs for:
  - `knowledge_domain_release_gate_status`
  - `knowledge_domain_release_gate_gate_open`
  - `knowledge_domain_release_gate_releasable_domains`
  - `knowledge_domain_release_gate_blocked_domains`
  - `knowledge_domain_release_gate_priority_domains`
  - `knowledge_domain_release_gate_blocking_reasons`
  - `knowledge_domain_release_gate_recommendations`
- Added job summary and downstream status-line coverage for release decision and
  runbook knowledge-domain release gate fields.

## Validation Commands
```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path('.github/workflows/evaluation-report.yml').read_text())
print('yaml-ok')
PY
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
git diff --check
```

## Expected Outcome
- The workflow can accept and surface release-gate JSON for release decision and
  release runbook.
- CI summary and workflow contract tests expose the new release-gate state
  without touching the dirty main worktree.
