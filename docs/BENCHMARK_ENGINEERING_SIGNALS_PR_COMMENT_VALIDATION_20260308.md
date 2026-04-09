# Benchmark Engineering Signals PR Comment Validation

## Goal

Expose benchmark engineering/knowledge signals directly in PR comments so
reviewers can see standards / tolerance / GD&T style evidence without opening
artifact JSON.

## Delivered

- Extended `Comment PR with results` in
  [evaluation-report.yml](../.github/workflows/evaluation-report.yml)
- Added PR comment variables and status lines for:
  - `benchmarkEngineeringEnabled`
  - `benchmarkEngineeringStatus`
  - `benchmarkEngineeringTopStandardTypes`
  - `benchmarkEngineeringStatusLine`
  - `benchmarkEngineeringLight`
- Added PR comment rows:
  - `Benchmark Engineering Signals`
  - `Benchmark Engineering Recommendations`
- Added signal light row:
  - `Benchmark Engineering Signals`

## Validation

Commands:

```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  --max-line-length=100

python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path('.github/workflows/evaluation-report.yml').read_text(encoding='utf-8'))
print('yaml_ok')
PY

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Result:

- `yaml_ok`
- `3 passed`
- `py_compile` passed
- `flake8` passed
