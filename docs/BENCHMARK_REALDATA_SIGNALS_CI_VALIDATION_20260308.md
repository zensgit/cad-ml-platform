# Benchmark Real-Data Signals CI Validation

## Scope

This delivery wires benchmark real-data signals into
`.github/workflows/evaluation-report.yml` so the workflow can:

- build `benchmark_realdata_signals`
- upload the generated artifact
- pass the generated JSON into downstream benchmark surfaces
- expose real-data status and recommendations in the job summary

## Workflow Changes

- Added `workflow_dispatch` inputs for real-data signal sources:
  - `benchmark_realdata_signals_enable`
  - `benchmark_realdata_signals_hybrid_summary_json`
  - `benchmark_realdata_signals_online_example_report_json`
  - `benchmark_realdata_signals_step_dir_summary_json`
- Added workflow env wiring for:
  - real-data exporter inputs
  - exporter outputs
  - downstream passthrough JSON hooks for bundle, companion, release decision, and runbook
- Added optional step:
  - `Build benchmark realdata signals (optional)`
- Added upload step:
  - `Upload benchmark realdata signals`
- Extended downstream benchmark build steps to accept
  `--benchmark-realdata-signals`
- Extended job summary with:
  - overall real-data status
  - component readiness counts
  - hybrid/history/STEP status lines
  - downstream bundle/companion/release/runbook real-data summaries

## Validation

Commands:

```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
python3 - <<'PY'
from pathlib import Path
import yaml
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text(encoding='utf-8'))
print('yaml-ok')
PY
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Results:

- `py_compile`: passed
- `flake8`: passed
- workflow YAML parse: passed
- `pytest`: `3 passed`

## Notes

- This change only wires CI and summary surfaces.
- PR comment support for real-data signals is intentionally left for the next stacked delivery.
