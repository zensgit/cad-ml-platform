# Benchmark Knowledge Source Coverage CI Validation

## Scope
- Wire `benchmark_knowledge_source_coverage` into `.github/workflows/evaluation-report.yml`
- Expose standalone artifact, summary lines, and downstream passthrough for:
  - `benchmark_artifact_bundle`
  - `benchmark_companion_summary`
  - `benchmark_release_decision`
  - `benchmark_release_runbook`
  - `benchmark_competitive_surpass_index`

## Key Changes
- Added `workflow_dispatch` input:
  - `benchmark_knowledge_source_coverage_enable`
- Added downstream override inputs:
  - `benchmark_competitive_surpass_index_knowledge_source_coverage_json`
  - `benchmark_artifact_bundle_knowledge_source_coverage_json`
  - `benchmark_companion_summary_knowledge_source_coverage_json`
  - `benchmark_release_decision_knowledge_source_coverage_json`
  - `benchmark_release_runbook_knowledge_source_coverage_json`
- Added env wiring for coverage artifact paths and downstream JSON overrides
- Added optional build step:
  - `Build benchmark knowledge source coverage (optional)`
- Added upload step:
  - `Upload benchmark knowledge source coverage`
- Added job summary lines for standalone coverage and downstream surfaces
- Added workflow contract assertions in:
  - `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation
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

## Result
- `yaml_ok`
- `py_compile` passed
- `flake8` passed
- `pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
  - `3 passed, 1 warning`
