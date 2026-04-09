# Benchmark Knowledge Domain Release Surface Alignment PR Comment Validation

## Scope
- PR: `feat/benchmark-knowledge-domain-release-alignment-pr-comment`
- Layer: `evaluation-report.yml` PR comment and signal lights
- Feature: `knowledge_domain_release_surface_alignment`

## Changes
- Added standalone PR comment variables for:
  - `benchmarkKnowledgeDomainReleaseSurfaceAlignmentEnabled`
  - `benchmarkKnowledgeDomainReleaseSurfaceAlignmentStatus`
  - `benchmarkKnowledgeDomainReleaseSurfaceAlignmentSummary`
  - `benchmarkKnowledgeDomainReleaseSurfaceAlignmentMismatches`
  - `benchmarkKnowledgeDomainReleaseSurfaceAlignmentArtifact`
- Added standalone status line and signal light.
- Added artifact bundle PR comment rows for:
  - `Benchmark Artifact Bundle Knowledge Domain Release Surface Alignment`
  - `Benchmark Artifact Bundle Knowledge Domain Release Surface Mismatches`
- Added companion summary PR comment rows for:
  - `Benchmark Companion Knowledge Domain Release Surface Alignment`
  - `Benchmark Companion Knowledge Domain Release Surface Mismatches`
- Extended workflow contract tests for new vars, rows, and light output.

## Validation
```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
python3 - <<'PY'
import yaml
from pathlib import Path
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text(encoding='utf-8'))
print('YAML_OK')
PY
git diff --check
```

## Result
- `py_compile`: passed
- `flake8`: passed
- `pytest`: passed
- workflow YAML parse: passed
- `git diff --check`: passed
