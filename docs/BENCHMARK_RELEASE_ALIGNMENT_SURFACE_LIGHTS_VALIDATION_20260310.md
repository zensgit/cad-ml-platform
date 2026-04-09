## Objective

Expose operator-adoption release-surface alignment as first-class signal lights for:

- Benchmark Release Decision
- Benchmark Release Runbook

This completes the operator-adoption alignment path across:

- standalone exporter
- bundle / companion surfaces
- job summary
- PR comment
- signal-light table

## Changes

- Added release-decision alignment light:
  - `benchmarkReleaseDecisionReleaseSurfaceAlignmentLight`
- Added release-runbook alignment light:
  - `benchmarkReleaseRunbookReleaseSurfaceAlignmentLight`
- Extended PR comment / signal-light markdown rows:
  - `Benchmark Release Decision Release Surface Alignment`
  - `Benchmark Release Runbook Release Surface Alignment`
- Extended workflow regression assertions for the new rows and payload bindings.

## Validation

Commands:

```bash
python3 - <<'PY'
from pathlib import Path
import yaml
yaml.safe_load(Path('.github/workflows/evaluation-report.yml').read_text())
print('yaml_ok')
PY

python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Expected result:

- workflow YAML parses successfully
- targeted workflow regression passes
- release decision / runbook alignment rows appear in PR comment coverage

## Outcome

Operator-adoption release-surface alignment is now represented consistently at the
signal-light layer for both release-decision and release-runbook benchmark
surfaces.
