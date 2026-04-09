# Benchmark Operator Outcome Drift Bundle Companion CI Validation

## Goal

Wire `operator_adoption_knowledge_outcome_drift` for benchmark artifact bundle and
benchmark companion summary into:

- `evaluation-report.yml` step outputs
- GitHub job summary
- PR comment / signal lights

## Changes

- Updated `.github/workflows/evaluation-report.yml` to export:
  - `operator_adoption_knowledge_outcome_drift_status`
  - `operator_adoption_knowledge_outcome_drift_summary`
  for both:
  - `benchmark_artifact_bundle`
  - `benchmark_companion_summary`
- Added job summary lines for both surfaces.
- Added PR comment table rows:
  - `Benchmark Artifact Bundle Operator Outcome Drift`
  - `Benchmark Companion Operator Outcome Drift`
- Extended workflow contract tests in
  `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`.

## Validation

```bash
python3 - <<'PY'
from pathlib import Path
import yaml
yaml.safe_load(Path('.github/workflows/evaluation-report.yml').read_text(encoding='utf-8'))
print('yaml-ok')
PY

python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Result:
- `yaml-ok`
- `3 passed`
- `1 warning`

## Outcome

- Bundle and companion operator outcome drift are now visible in CI and PR review surfaces,
  not only in downstream JSON/Markdown artifacts.
