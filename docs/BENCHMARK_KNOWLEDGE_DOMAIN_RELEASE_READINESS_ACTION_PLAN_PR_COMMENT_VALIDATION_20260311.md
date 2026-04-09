# Benchmark Knowledge Domain Release Readiness Action Plan PR Comment Validation

## Scope
- Add `knowledge_domain_release_readiness_action_plan` to PR comment and signal lights.
- Surface downstream status lines for:
  - artifact bundle
  - companion summary
  - release decision
  - release runbook
- Repair the existing `benchmarkKnowledgeDomainReleaseGateLight` / `benchmarkKnowledgeDomainReleaseSurfaceAlignmentLight` JavaScript block so the PR comment script remains syntactically coherent.

## Files
- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation
```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
python3 - <<'PY'
import pathlib
import yaml
path = pathlib.Path(".github/workflows/evaluation-report.yml")
yaml.safe_load(path.read_text(encoding="utf-8"))
print("ok")
PY
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
git diff --check
```

## Expected
- PR comment contains:
  - `Benchmark Knowledge Domain Release Readiness Action Plan`
  - `Benchmark Artifact Bundle Knowledge Domain Release Readiness Action Plan`
  - `Benchmark Companion Knowledge Domain Release Readiness Action Plan`
  - `Benchmark Release Decision Knowledge Domain Release Readiness Action Plan`
  - `Benchmark Release Runbook Knowledge Domain Release Readiness Action Plan`
- Signal lights contain:
  - `Benchmark Knowledge Domain Release Readiness Action Plan`
- Workflow test suite covers the new PR comment wiring.
