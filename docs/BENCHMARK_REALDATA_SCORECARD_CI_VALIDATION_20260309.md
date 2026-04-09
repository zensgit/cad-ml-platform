# Benchmark Real-Data Scorecard CI Validation

## Goal
- Wire `benchmark_realdata_scorecard` into `evaluation-report.yml`.
- Propagate scorecard artifacts into companion, bundle, release decision, and release runbook surfaces.
- Keep the workflow test contract aligned with the new dispatch inputs, build step, downstream args, and upload step.

## Key Changes
- Added workflow dispatch inputs for:
  - `benchmark_realdata_scorecard_enable`
  - `benchmark_realdata_scorecard_hybrid_summary_json`
  - `benchmark_realdata_scorecard_history_summary_json`
  - `benchmark_realdata_scorecard_online_example_report_json`
  - `benchmark_realdata_scorecard_step_dir_summary_json`
- Added downstream passthrough inputs for:
  - `benchmark_artifact_bundle_realdata_scorecard_json`
  - `benchmark_companion_summary_realdata_scorecard_json`
  - `benchmark_release_decision_realdata_scorecard_json`
  - `benchmark_release_runbook_realdata_scorecard_json`
- Added workflow env vars and output paths for `benchmark_realdata_scorecard`.
- Added `Build benchmark real-data scorecard (optional)`.
- Passed `--benchmark-realdata-scorecard` into:
  - `export_benchmark_artifact_bundle.py`
  - `export_benchmark_companion_summary.py`
  - `export_benchmark_release_decision.py`
  - `export_benchmark_release_runbook.py`
- Added `Upload benchmark realdata scorecard`.
- Extended workflow tests to assert the new contract.

## Files
- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation
```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
python3 - <<'PY'
import yaml
from pathlib import Path
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text(encoding='utf-8'))
print('yaml_ok')
PY
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
pytest -q \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/unit/test_benchmark_realdata_scorecard.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py
```

## Result
- `py_compile`: passed
- workflow YAML parse: passed
- `flake8`: passed
- `pytest`: `24 passed, 1 warning`
