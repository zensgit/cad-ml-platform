# Benchmark Knowledge Domain Drift CI Validation

## Scope

Wire the new benchmark knowledge domain drift fields into workflow outputs, job
summary, and PR comment surfaces.

## Delivered

- `benchmark_knowledge_drift` step now exports:
  - `domain_regressions`
  - `domain_improvements`
  - `resolved_priority_domains`
  - `new_priority_domains`
- benchmark artifact bundle / companion / release decision / release runbook
  workflow parse blocks now expose the same domain-drift fields from their
  nested `knowledge_drift` payloads
- PR comment status lines now include domain-drift state for benchmark,
  artifact bundle, companion, release decision, and release runbook surfaces
- job summary now emits domain-drift regressions/improvements/resolved/new
  domains across those same surfaces

## Validation

Commands:

```bash
python3 - <<'PY'
import yaml, pathlib
p = pathlib.Path(".github/workflows/evaluation-report.yml")
yaml.safe_load(p.read_text())
print("yaml-ok")
PY

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Result:

- YAML parse passed
- `flake8` passed
- `pytest` passed: `3 passed`
