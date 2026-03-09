# Benchmark Knowledge Outcome Correlation PR Comment Validation

## Goal

Expose `benchmark_knowledge_outcome_correlation` in the PR comment and signal
light surfaces so reviewers can see:

- the top-level outcome-correlation status
- outcome-correlation recommendations
- downstream bundle / companion / release / runbook status lines
- a dedicated signal light row

## Implementation

Primary files:

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Added PR Comment Coverage

The PR comment now includes:

- `Benchmark Knowledge Outcome Correlation`
- `Benchmark Knowledge Outcome Recommendations`
- `Benchmark Artifact Bundle Knowledge Outcome Correlation`
- `Benchmark Companion Knowledge Outcome Correlation`
- `Benchmark Release Decision Knowledge Outcome Correlation`
- `Benchmark Release Runbook Knowledge Outcome Correlation`

The signal-light block now includes:

- `Benchmark Knowledge Outcome Correlation`

The workflow script now exports and renders:

- `benchmarkKnowledgeOutcomeCorrelationStatusLine`
- `benchmarkKnowledgeOutcomeCorrelationLight`
- downstream `KnowledgeOutcomeCorrelationStatusLine` rows for
  bundle / companion / release / runbook

## Validation Commands

```bash
python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path('.github/workflows/evaluation-report.yml').read_text())
print('yaml_ok')
PY

python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

git diff --check
```

## Validation Result

- workflow YAML parse: pass
- `py_compile`: pass
- `flake8`: pass
- `pytest`: `3 passed`
- `git diff --check`: pass

## Outcome

`knowledge_outcome_correlation` now reaches the same reviewer-facing surfaces as
the other benchmark knowledge dimensions: CI artifact, job summary, PR comment,
and signal lights. This completes the stacked PR comment layer for the
knowledge-outcome-correlation benchmark line.
