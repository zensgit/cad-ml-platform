# Benchmark Knowledge Real-Data PR Comment Validation

## Scope
- Extend `evaluation-report.yml` PR comment / signal lights with
  `knowledge_realdata_correlation`
- Mirror existing benchmark patterns for:
  - top-level benchmark status rows
  - artifact bundle
  - companion summary
  - release decision
  - release runbook

## Key Changes
- Added PR comment JS bindings for:
  - `benchmarkKnowledgeRealdataCorrelation*`
  - `benchmarkArtifactBundleKnowledgeRealdataCorrelation*`
  - `benchmarkCompanionKnowledgeRealdataCorrelation*`
  - `benchmarkReleaseKnowledgeRealdataCorrelation*`
  - `benchmarkReleaseRunbookKnowledgeRealdataCorrelation*`
- Added status line renderers for all five surfaces
- Added top-level signal light:
  - `benchmarkKnowledgeRealdataCorrelationLight`
- Added PR comment rows:
  - `Benchmark Knowledge Real-Data Correlation`
  - `Benchmark Knowledge Real-Data Recommendations`
  - `Benchmark Artifact Bundle Knowledge Real-Data`
  - `Benchmark Companion Knowledge Real-Data`
  - `Benchmark Release Decision Knowledge Real-Data`
  - `Benchmark Release Runbook Knowledge Real-Data`

## Validation
Commands run in isolated worktree:

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
```

Results:
- YAML parse: passed
- `py_compile`: passed
- `flake8`: passed
- `pytest`: `3 passed`

## Outcome
`knowledge_realdata_correlation` is now visible in the same operator-facing surfaces
as knowledge application and real-data signals:
- job summary
- PR comment status tables
- signal lights

That keeps the benchmark stack consistent for review and release decisions.
