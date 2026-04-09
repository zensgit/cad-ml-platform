# Benchmark Knowledge Domain Action Plan CI Validation

Date: 2026-03-10

## Scope

This change wires `benchmark_knowledge_domain_action_plan` into the benchmark
CI control plane:

- workflow_dispatch inputs
- workflow environment variables
- standalone exporter step
- downstream bundle / companion / release / runbook passthrough
- artifact upload
- job summary outputs

## Files

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation

```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
python3 - <<'PY'
from pathlib import Path
import yaml
yaml.safe_load(Path('.github/workflows/evaluation-report.yml').read_text())
print('yaml_ok')
PY
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Results:

- `py_compile` passed
- `flake8` passed
- workflow YAML parse passed
- `pytest` passed: `3 passed`

## Notes

- The release-decision embedded output writer now emits:
  - `knowledge_domain_action_plan_status`
  - `knowledge_domain_action_plan_actions`
  - `knowledge_domain_action_plan_priority_domains`
  - `knowledge_domain_action_plan_recommendations`
- This closes the gap between the new standalone exporter and downstream
  benchmark workflow contracts.
