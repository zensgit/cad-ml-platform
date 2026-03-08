# Benchmark Knowledge Domain Matrix PR Comment Validation

## Goal

Expose `knowledge_domain_matrix` in the GitHub PR comment / signal-light layer so
the new benchmark domain surface is visible alongside:

- knowledge readiness
- knowledge drift
- knowledge application
- knowledge real-data correlation

## Scope

Updated:

- [.github/workflows/evaluation-report.yml](../.github/workflows/evaluation-report.yml)
- [test_evaluation_report_workflow_graph2d_extensions.py](../tests/unit/test_evaluation_report_workflow_graph2d_extensions.py)

## Design

Added `knowledge_domain_matrix` comment support in three places:

1. Workflow step outputs -> JavaScript bindings
2. PR comment status-line rendering
3. Signal-light table rendering

New PR comment data includes:

- top-level matrix status line
- bundle matrix status line
- companion matrix status line
- release decision matrix status line
- release runbook matrix status line

New JS bindings include:

- `benchmarkKnowledgeDomainMatrix*`
- `benchmarkArtifactBundleKnowledgeDomainMatrix*`
- `benchmarkCompanionKnowledgeDomainMatrix*`
- `benchmarkReleaseKnowledgeDomainMatrix*`
- `benchmarkReleaseRunbookKnowledgeDomainMatrix*`

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

- This branch only updates PR comment / signal lights.
- It assumes the stacked CI branch is present underneath so matrix outputs are
  already available from workflow steps.
