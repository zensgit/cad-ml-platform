# Benchmark Knowledge Outcome Correlation CI Validation

## Goal

Wire `benchmark_knowledge_outcome_correlation` into `evaluation-report.yml` so the
workflow can:

- accept manual/dispatch inputs
- build the exporter artifact
- upload the standalone artifact
- pass the new JSON into bundle / companion / release decision / release runbook

## Implementation

Primary files:

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Added Workflow Coverage

Dispatch inputs:

- `benchmark_knowledge_outcome_correlation_enable`
- `benchmark_knowledge_outcome_correlation_knowledge_domain_matrix_json`
- `benchmark_knowledge_outcome_correlation_realdata_scorecard_json`
- downstream surface passthrough inputs:
  - `benchmark_artifact_bundle_knowledge_outcome_correlation_json`
  - `benchmark_companion_summary_knowledge_outcome_correlation_json`
  - `benchmark_release_decision_knowledge_outcome_correlation_json`
  - `benchmark_release_runbook_knowledge_outcome_correlation_json`

Workflow changes:

- `Build benchmark knowledge outcome correlation (optional)`
- `Upload benchmark knowledge outcome correlation`
- bundle / companion / release / runbook steps now pass
  `--benchmark-knowledge-outcome-correlation`
- bundle / companion / release / runbook outputs now expose:
  - `knowledge_outcome_correlation_status`
  - `knowledge_outcome_correlation_focus_areas`
  - `knowledge_outcome_correlation_priority_domains`
  - `knowledge_outcome_correlation_domain_statuses`
  - `knowledge_outcome_correlation_recommendations`

## Validation Commands

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

## Validation Result

- workflow YAML parse: pass
- `py_compile`: pass
- `flake8`: pass
- `pytest`: `3 passed`

## Outcome

`knowledge_outcome_correlation` is now a first-class CI artifact and a reusable
input for the benchmark release-support surfaces. The next stacked step is PR
comment / signal-light wiring.
