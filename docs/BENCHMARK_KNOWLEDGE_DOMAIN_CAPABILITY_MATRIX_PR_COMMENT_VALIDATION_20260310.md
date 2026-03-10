# Benchmark Knowledge Domain Capability Matrix PR Comment Validation

## Goal

Expose `knowledge_domain_capability_matrix` in PR-facing benchmark surfaces so
reviewers can see standards / tolerance / GD&T capability gaps directly from the
GitHub comment and signal-light sections.

## Changes

- Added PR comment variables for:
  - top-level `knowledge_domain_capability_matrix`
  - artifact bundle passthrough
  - companion summary passthrough
  - release decision passthrough
  - release runbook passthrough
- Added status-line formatting for:
  - ready / partial / blocked / total domains
  - focus areas
  - priority domains
  - provider gaps
  - surface gaps
  - recommendations
- Added `Benchmark Knowledge Domain Capability Matrix` row to:
  - Additional Analysis table
  - downstream bundle / companion / release / runbook sections
  - signal-lights table
- Extended workflow regression tests to assert the new variables, rows, and
  recommendation rendering.

## Validation

Commands run in `/private/tmp/cad-ml-platform-knowledge-domain-capability-pr-comment-20260310`:

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

Capability-matrix gaps for `tolerance`, `standards`, and `gdt` are now visible
from the PR review surface instead of being hidden only inside standalone benchmark
artifacts.
