# Benchmark Artifact Bundle Companion CI Validation

## Goal

Ensure `evaluation-report.yml` passes benchmark companion summary outputs into
`scripts/export_benchmark_artifact_bundle.py`.

## Changes

- Added workflow-dispatch input:
  - `benchmark_artifact_bundle_companion_summary_json`
- Added environment variable:
  - `BENCHMARK_ARTIFACT_BUNDLE_COMPANION_SUMMARY_JSON`
- Updated `Build benchmark artifact bundle (optional)` to include:
  - workflow input override
  - `steps.benchmark_companion_summary.outputs.output_json`
  - environment fallback

## Validation

```bash
python3 -m py_compile scripts/export_benchmark_artifact_bundle.py tests/unit/test_benchmark_artifact_bundle.py tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 scripts/export_benchmark_artifact_bundle.py tests/unit/test_benchmark_artifact_bundle.py tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
pytest -q tests/unit/test_benchmark_artifact_bundle.py tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
python3 - <<'PY'
from pathlib import Path
import yaml
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text(encoding='utf-8'))
print('yaml_ok')
PY
```

## Expected Outcome

- Benchmark artifact bundle generation automatically absorbs benchmark companion summary
  whenever that artifact exists in the same workflow run.
