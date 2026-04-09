# Benchmark Engineering Signals CI Validation

## Goal

Wire benchmark engineering signals into `evaluation-report.yml` so workflow runs
can build, upload, and surface engineering-semantic evidence alongside the
existing benchmark scorecard.

## Delivered

- Added workflow-dispatch inputs:
  - `benchmark_scorecard_engineering_signals_summary`
  - `benchmark_engineering_signals_enable`
  - `benchmark_engineering_signals_hybrid_summary_json`
  - `benchmark_engineering_signals_ocr_review_summary_json`
- Added env configuration:
  - `BENCHMARK_SCORECARD_ENGINEERING_SIGNALS_SUMMARY_JSON`
  - `BENCHMARK_ENGINEERING_SIGNALS_*`
- Added workflow step:
  - `Build benchmark engineering signals (optional)`
- Added artifact upload step:
  - `Upload benchmark engineering signals`
- Extended benchmark scorecard generation so workflow runs can consume:
  - explicit engineering summary input
  - workflow-produced engineering summary artifact
- Added job-summary lines for:
  - engineering status
  - engineering coverage ratio
  - engineering standard types
  - engineering artifact path

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

## Notes

- This change wires engineering signals into job summary and artifact upload.
- PR comment wiring remains a separate follow-up so the stack stays low-risk.
