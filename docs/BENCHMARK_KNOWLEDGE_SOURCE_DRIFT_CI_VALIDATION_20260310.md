# Benchmark Knowledge Source Drift CI Validation

## Scope
- Add `benchmark_knowledge_source_drift` workflow dispatch inputs and environment wiring.
- Build and upload standalone `knowledge_source_drift` artifacts in `evaluation-report.yml`.
- Pass `knowledge_source_drift` JSON into downstream benchmark surfaces:
  - artifact bundle
  - companion summary
  - release decision
  - release runbook
- Expose `knowledge_source_drift_*` step outputs for CI summary consumption.

## Key Changes
- Updated [.github/workflows/evaluation-report.yml](/private/tmp/cad-ml-platform-knowledge-source-drift-ci-iN9iLo/.github/workflows/evaluation-report.yml)
  - Added workflow inputs:
    - `benchmark_knowledge_source_drift_enable`
    - `benchmark_knowledge_source_drift_current_summary_json`
    - `benchmark_knowledge_source_drift_previous_summary_json`
  - Added environment variables:
    - `BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_*`
  - Added step:
    - `Build benchmark knowledge source drift (optional)`
  - Added artifact upload step:
    - `Upload benchmark knowledge source drift`
  - Added CI summary rows for standalone and downstream `knowledge_source_drift` status.
- Updated [test_evaluation_report_workflow_graph2d_extensions.py](/private/tmp/cad-ml-platform-knowledge-source-drift-ci-iN9iLo/tests/unit/test_evaluation_report_workflow_graph2d_extensions.py)
  - Dispatch input assertions
  - Build step assertions
  - Downstream benchmark script assertions
  - Upload step assertion

## Validation
Commands run in `/private/tmp/cad-ml-platform-knowledge-source-drift-ci-iN9iLo`:

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
- `py_compile`: passed
- `flake8`: passed
- workflow YAML parse: passed
- `pytest`: `3 passed, 1 warning`
- `git diff --check`: passed

## Notes
- This branch only wires the CI/build layer and downstream step outputs.
- PR comment / signal-light wiring should stack on top of this branch to reuse the new step outputs directly.
