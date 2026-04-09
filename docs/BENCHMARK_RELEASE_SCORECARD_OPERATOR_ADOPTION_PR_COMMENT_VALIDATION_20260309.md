# Benchmark Release Scorecard Operator Adoption PR Comment Validation

## Scope

This change extends the benchmark PR comment and signal lights so release
decision and release runbook surfaces expose scorecard and operational
operator-adoption state alongside the already wired release-level operator
drift signals.

Added PR comment rows:

- `Benchmark Release Decision Scorecard Operator Adoption`
- `Benchmark Release Decision Scorecard Operator Outcome Drift`
- `Benchmark Release Decision Operational Operator Adoption`
- `Benchmark Release Decision Operational Operator Outcome Drift`
- `Benchmark Release Runbook Scorecard Operator Adoption`
- `Benchmark Release Runbook Scorecard Operator Outcome Drift`
- `Benchmark Release Runbook Operational Operator Adoption`
- `Benchmark Release Runbook Operational Operator Outcome Drift`

Added signal lights:

- `benchmarkReleaseDecisionScorecardOperatorAdoptionLight`
- `benchmarkReleaseDecisionScorecardOperatorOutcomeDriftLight`
- `benchmarkReleaseDecisionOperationalOperatorAdoptionLight`
- `benchmarkReleaseDecisionOperationalOperatorOutcomeDriftLight`
- `benchmarkReleaseRunbookScorecardOperatorAdoptionLight`
- `benchmarkReleaseRunbookScorecardOperatorOutcomeDriftLight`
- `benchmarkReleaseRunbookOperationalOperatorAdoptionLight`
- `benchmarkReleaseRunbookOperationalOperatorOutcomeDriftLight`

## Files

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation

Executed:

```bash
python3 - <<'PY'
from pathlib import Path
import yaml
yaml.safe_load(Path(".github/workflows/evaluation-report.yml").read_text())
print("yaml-ok")
PY

python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Expected result:

- workflow YAML parses cleanly
- workflow regression assertions cover the new release decision / runbook
  scorecard and operational operator-adoption rows
- PR comment contract remains stable after CI wiring from `#282`
