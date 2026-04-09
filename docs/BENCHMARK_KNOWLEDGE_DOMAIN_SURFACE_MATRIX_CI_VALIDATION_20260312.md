# Benchmark Knowledge Domain Surface Matrix CI Validation

## Goal

Wire `knowledge_domain_surface_matrix` into `evaluation-report.yml` so the
benchmark component is built, uploaded, summarized, and forwarded into
downstream bundle / companion / release surfaces.

## Scope

- Added workflow-dispatch inputs and env defaults for
  `benchmark_knowledge_domain_surface_matrix`
- Added optional build step:
  `Build benchmark knowledge domain surface matrix (optional)`
- Added artifact upload step
- Added job-summary lines for status, domain counts, public surface gaps,
  reference gaps, and recommendations
- Forwarded generated JSON into:
  - benchmark artifact bundle
  - benchmark companion summary
  - benchmark release decision
  - benchmark release runbook
- Added workflow regression coverage

## Validation

```bash
python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path(".github/workflows/evaluation-report.yml").read_text())
print("yaml_ok")
PY

python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  -k 'knowledge_domain_surface_matrix or knowledge_domain_api_surface_matrix'
```

## Expected Result

- Workflow YAML parses successfully
- CI wiring test finds:
  - dispatch inputs
  - env defaults
  - build step
  - upload step
  - job-summary lines
- Existing `knowledge_domain_api_surface_matrix` workflow wiring remains intact
