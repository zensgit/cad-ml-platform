# Benchmark Knowledge Domain Readiness CI Validation

Date: 2026-03-08

## Goal

Wire `knowledge domain readiness` through benchmark CI surfaces so the workflow,
job summary, and PR comment expose benchmark-facing domain gaps instead of only
component-level focus areas.

## Scope

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Delivered

- `benchmark_knowledge_readiness` step now exports:
  - `domain_count`
  - `priority_domains`
  - `domain_focus_areas`
- `benchmark_artifact_bundle`, `benchmark_companion_summary`,
  `benchmark_release_decision`, and `benchmark_release_runbook` steps now export:
  - `knowledge_priority_domains`
  - `knowledge_domain_focus_areas`
- Job summary now includes knowledge-domain lines for readiness and the main
  downstream benchmark release surfaces.
- PR comment / signal-light template now includes knowledge-domain visibility in:
  - benchmark knowledge status
  - artifact bundle status
  - companion summary status
  - release decision status
  - release runbook status
  - additional analysis rows

## Validation

```bash
python3 - <<'PY'
import yaml, pathlib
p = pathlib.Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(p.read_text())
print('ok')
PY

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

git diff --check
```

## Result

- YAML parse: passed
- `flake8`: passed
- `pytest`: `3 passed`
- `git diff --check`: passed

## Notes

- This change intentionally keeps knowledge-domain visibility additive.
- Existing component-level `knowledge_focus_areas` and `knowledge_drift` outputs
  remain intact for backward-compatible review/report consumers.
