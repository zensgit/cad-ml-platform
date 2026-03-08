# Benchmark Knowledge Real-Data CI Validation

## Scope
- Wire `benchmark_knowledge_realdata_correlation` into `evaluation-report.yml`
- Propagate correlation JSON into:
  - artifact bundle
  - companion summary
  - release decision
  - release runbook
- Expose correlation status in:
  - workflow artifacts
  - job summary
  - workflow tests

## Key Changes
- Added workflow dispatch inputs for:
  - `benchmark_knowledge_realdata_correlation_enable`
  - `benchmark_knowledge_realdata_correlation_knowledge_readiness_json`
  - `benchmark_knowledge_realdata_correlation_knowledge_application_json`
  - `benchmark_knowledge_realdata_correlation_realdata_signals_json`
- Added env passthroughs for downstream surfaces:
  - `BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_REALDATA_CORRELATION_JSON`
  - `BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_REALDATA_CORRELATION_JSON`
  - `BENCHMARK_RELEASE_DECISION_KNOWLEDGE_REALDATA_CORRELATION_JSON`
  - `BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_REALDATA_CORRELATION_JSON`
- Added workflow step:
  - `Build benchmark knowledge realdata correlation (optional)`
- Added artifact upload step:
  - `Upload benchmark knowledge realdata correlation`
- Extended downstream step outputs for:
  - artifact bundle
  - companion summary
  - release decision
  - release runbook
- Extended job summary with:
  - top-level correlation status
  - ready / partial / blocked / total domains
  - focus areas
  - priority domains
  - recommendations
  - surface-level correlation lines

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
`knowledge_realdata_correlation` now has a first-class CI path comparable to:
- `knowledge_application`
- `realdata_signals`
- downstream benchmark release surfaces

This keeps the benchmark stack aligned with the broader competitive-surpass direction:
- knowledge gaps are no longer standalone exporter output only
- real-data linkage is now observable in CI and downstream release surfaces
