# Benchmark Knowledge Domain Matrix CI Validation

## Goal

Wire the new `knowledge_domain_matrix` benchmark artifact into the shared
evaluation workflow so it behaves like the existing benchmark layers:

- optional build step
- artifact upload
- downstream benchmark surface passthrough
- job summary exposure

## Scope

Updated:

- [.github/workflows/evaluation-report.yml](../.github/workflows/evaluation-report.yml)
- [test_evaluation_report_workflow_graph2d_extensions.py](../tests/unit/test_evaluation_report_workflow_graph2d_extensions.py)

## Design

### Standalone workflow support

Added new workflow inputs and env vars for:

- enable override
- readiness JSON
- application JSON
- real-data correlation JSON
- output JSON / Markdown paths

Added new optional step:

- `Build benchmark knowledge domain matrix (optional)`

Added new upload step:

- `Upload benchmark knowledge domain matrix`

### Downstream surface passthrough

Extended these workflow build steps to accept
`--benchmark-knowledge-domain-matrix`:

- artifact bundle
- companion summary
- release decision
- release runbook

The workflow now exports downstream matrix fields:

- `knowledge_domain_matrix_status`
- `knowledge_domain_matrix_focus_areas`
- `knowledge_domain_matrix_priority_domains`
- `knowledge_domain_matrix_domain_statuses`
- `knowledge_domain_matrix_recommendations`

### Job summary

Added summary lines for:

- standalone matrix status
- bundle matrix status
- companion matrix status
- release decision matrix status
- release runbook matrix status

## Validation

Commands run in isolated worktree
`/private/tmp/cad-ml-platform-knowledge-realdata-correlation-20260309`.

```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  --max-line-length=100

python3 - <<'PY'
import yaml
from pathlib import Path
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text())
print('yaml_ok')
PY

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Results:

- `py_compile`: pass
- `flake8`: pass
- YAML parse: `yaml_ok`
- `pytest`: `3 passed`

## Limitations

- This branch wires CI and job summary only.
- PR comment / signal light integration is intentionally left for the stacked
  follow-up branch.
