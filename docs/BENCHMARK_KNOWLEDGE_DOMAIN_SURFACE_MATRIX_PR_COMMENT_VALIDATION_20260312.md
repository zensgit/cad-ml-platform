# Benchmark Knowledge Domain Surface Matrix PR Comment Validation

## Goal

Expose `knowledge_domain_surface_matrix` in PR comments and signal lights so the
benchmark control-plane surfaces are visible during review.

## Scope

- Added PR-comment script bindings for:
  - `benchmarkKnowledgeDomainSurfaceMatrixEnabled`
  - `benchmarkKnowledgeDomainSurfaceMatrixStatusLine`
  - `benchmarkKnowledgeDomainSurfaceMatrixLight`
  - `benchmarkKnowledgeDomainSurfaceMatrixPublicSurfaceGapDomains`
  - `benchmarkKnowledgeDomainSurfaceMatrixReferenceGapDomains`
- Added review table row:
  - `Benchmark Knowledge Domain Surface Matrix`
- Added signal-light row:
  - `Benchmark Knowledge Domain Surface Matrix`

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
- PR comment regression finds the new `knowledge_domain_surface_matrix` bindings
- Existing `knowledge_domain_api_surface_matrix` PR comment behavior remains intact
