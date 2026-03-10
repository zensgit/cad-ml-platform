# Benchmark Knowledge Domain Capability Matrix CI Validation

## Goal

Wire `knowledge_domain_capability_matrix` into the benchmark control-plane so the
workflow can build it, upload it, surface it in job summary output, and pass it
through to downstream benchmark surfaces.

## Changes

- Added `workflow_dispatch` inputs for:
  - `benchmark_knowledge_domain_capability_matrix_enable`
  - `benchmark_knowledge_domain_capability_matrix_knowledge_readiness_json`
  - `benchmark_knowledge_domain_capability_matrix_knowledge_application_json`
  - `benchmark_knowledge_domain_capability_matrix_knowledge_domain_matrix_json`
  - downstream JSON overrides for bundle / companion / release decision / runbook
- Added CI env defaults for the capability-matrix exporter and downstream inputs
- Added workflow step:
  - `Build benchmark knowledge domain capability matrix (optional)`
- Added artifact upload step:
  - `Upload benchmark knowledge domain capability matrix`
- Added job summary lines for:
  - capability-matrix status and counts
  - provider gaps / surface gaps
  - downstream bundle / companion / release decision / release runbook status
- Extended workflow regression tests to cover:
  - inputs and env
  - build step script and emitted outputs
  - upload step gate
  - downstream exporter passthrough
  - job summary rendering

## Validation

Commands run in `/private/tmp/cad-ml-platform-knowledge-domain-capability-ci-20260310`:

```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
python3 - <<'PY'
from pathlib import Path
import yaml
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text())
print('yaml_ok')
PY
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Results:

- `py_compile` passed
- `flake8` passed
- `yaml.safe_load(...)` passed
- `pytest` passed: `3 passed, 1 warning`

## Outcome

`knowledge_domain_capability_matrix` is now wired through CI at the same layer as
other benchmark knowledge surfaces, and the benchmark control-plane can expose
capability gaps for `standards`, `tolerance`, and `gdt` consistently in artifacts
and summaries.
