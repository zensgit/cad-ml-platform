# Benchmark Knowledge Domain Release Gate PR Comment Validation

## Scope
- Extend `evaluation-report.yml` PR comment and signal lights with `knowledge_domain_release_gate`
- Surface standalone, artifact bundle, and companion release-gate status lines
- Add workflow-contract assertions for the new PR comment strings and variables

## Key Changes
- Added PR comment state variables for:
  - `benchmarkKnowledgeDomainReleaseGate*`
  - `benchmarkArtifactBundleKnowledgeDomainReleaseGate*`
  - `benchmarkCompanionKnowledgeDomainReleaseGate*`
- Added status-line renderers for standalone, bundle, and companion release-gate outputs
- Added a new signal light:
  - `Benchmark Knowledge Domain Release Gate`
- Added PR comment rows for:
  - `Benchmark Knowledge Domain Release Gate`
  - `Benchmark Artifact Bundle Knowledge Domain Release Gate`
  - `Benchmark Companion Knowledge Domain Release Gate`
- Extended workflow tests to assert the new variables, table rows, and signal-light entry

## Validation
```bash
python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path(".github/workflows/evaluation-report.yml").read_text())
print("yaml-ok")
PY

python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

git diff --check
```

## Result
- Workflow YAML parsed successfully
- `py_compile` passed
- `flake8` passed
- `pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py` passed
- `git diff --check` passed
