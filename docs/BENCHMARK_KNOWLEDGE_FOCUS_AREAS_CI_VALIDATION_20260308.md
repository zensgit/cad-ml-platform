# Benchmark Knowledge Focus Areas CI Validation

## Goal

Expose benchmark knowledge focus areas through CI so operators and reviewers can
see which knowledge domains remain weak without opening raw JSON artifacts.

## Design

- Extend `evaluation-report.yml` benchmark outputs with knowledge focus metadata:
  - scorecard:
    - `knowledge_focus_area_count`
    - `knowledge_focus_areas`
  - standalone knowledge readiness:
    - `focus_area_count`
    - `focus_areas`
  - artifact bundle:
    - `knowledge_focus_areas`
  - companion summary:
    - `knowledge_focus_areas`
  - release decision:
    - `knowledge_focus_areas`
  - release runbook:
    - `knowledge_focus_areas`
- Add knowledge focus lines to:
  - job summary
  - PR comment

## Files

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation

Commands:

```bash
python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path('.github/workflows/evaluation-report.yml').read_text(encoding='utf-8'))
print('yaml_ok')
PY

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Results:

- YAML parse: `yaml_ok`
- `flake8`: pass
- `pytest`: `3 passed, 1 warning`

## Outcome

- CI can now report not only `knowledge_foundation_partial`, but also the
  compact focus-area breakdown that explains where the remaining knowledge gaps
  live.
- PR comments and job summaries now surface the same focus-area signal used by
  scorecard, companion, release decision, and release runbook.
