# Benchmark Knowledge Source Drift PR Comment Validation

## Scope
- Wire `benchmark_knowledge_source_drift` into the `evaluation-report.yml` PR comment table.
- Add a dedicated signal light for top-level benchmark knowledge source drift.
- Surface downstream status lines for:
  - artifact bundle
  - companion summary
  - release decision
  - release runbook

## Key Changes
- Added PR comment variables for standalone `knowledge_source_drift` step outputs.
- Added downstream PR comment variables for bundle, companion, release decision, and release runbook outputs.
- Added `benchmarkKnowledgeSourceDriftStatusLine`.
- Added `benchmarkKnowledgeSourceDriftLight`.
- Added new PR comment rows for benchmark and downstream knowledge source drift surfaces.
- Extended workflow contract assertions in `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`.

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
git diff --check
```

## Result
- Workflow YAML parses successfully.
- Workflow contract test covers knowledge source drift PR comment and signal-light wiring.
- No whitespace or patch formatting errors remain.
