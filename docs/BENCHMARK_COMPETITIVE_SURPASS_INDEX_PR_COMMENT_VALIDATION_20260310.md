# Benchmark Competitive Surpass Index PR Comment Validation

## Scope

This change extends the evaluation workflow PR comment so
`benchmark_competitive_surpass_index` appears in the same review surfaces as the
other benchmark pillars.

Integrated areas:

- PR comment status table
- Graph2D signal lights table
- downstream bundle/companion PR comment rows

## Design

The PR comment layer now reads three competitive-surpass views:

1. standalone `benchmark_competitive_surpass_index`
2. `benchmark artifact bundle` passthrough
3. `benchmark companion summary` passthrough

The PR comment exposes:

- standalone status, score, ready/partial/blocked pillars, gaps, recommendations
- bundle status, primary gaps, recommendations
- companion status, primary gaps, recommendations

Signal lights are derived with the same semantics used elsewhere:

- `competitive_surpass_ready` -> green
- `competitive_surpass_blocked` -> red
- all other enabled states -> yellow
- disabled -> white

## Files

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation

Commands run:

```bash
python3 - <<'PY'
import yaml
from pathlib import Path
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text(encoding='utf-8'))
print('yaml_ok')
PY

python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Results:

- YAML parse: passed
- `py_compile`: passed
- `flake8`: passed
- `pytest`: pending local re-run after workflow patch

## Outcome

`competitive_surpass_index` is no longer summary-only. Reviewers will see the
standalone index plus bundle/companion competitive-surpass state directly in the
PR comment and signal lights.
