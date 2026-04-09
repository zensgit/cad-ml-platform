# Benchmark Knowledge Source Action Plan CI Validation

## Goal

Wire `benchmark_knowledge_source_action_plan` into the evaluation control plane so
the workflow can build, upload, summarize, and propagate knowledge-source action
plan signals into downstream benchmark surfaces.

## Scope

- Add `workflow_dispatch` inputs for standalone enablement and downstream JSON
  overrides.
- Add workflow env for `knowledge_source_action_plan`.
- Add `Build benchmark knowledge source action plan (optional)` step.
- Pass `--benchmark-knowledge-source-action-plan` into:
  - artifact bundle
  - companion summary
  - release decision
  - release runbook
  - competitive surpass index
- Export `knowledge_source_action_plan_*` outputs from downstream steps.
- Upload the standalone artifact.
- Add job summary lines for standalone and downstream surfaces.
- Extend workflow contract coverage in
  `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`.

## Key Files

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation

```bash
python3 - <<'PY'
from pathlib import Path
import yaml
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text(encoding='utf-8'))
print('yaml_ok')
PY

python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Result

- Workflow YAML parse: passed
- `py_compile`: passed
- `flake8`: passed
- `pytest`: `3 passed`

## Notes

- This branch only covers CI wiring.
- PR comment / signal-light exposure is intentionally left for the stacked
  follow-up branch.
