# Benchmark Competitive Surpass Index Knowledge Source Drift CI Validation

## Goal
- Wire `benchmark_knowledge_source_drift` into the `benchmark_competitive_surpass_index`
  workflow build path so CI can pass the new benchmark input through
  `evaluation-report.yml`.

## Changes
- Added workflow dispatch input:
  - `benchmark_competitive_surpass_index_knowledge_source_drift_json`
- Added workflow env passthrough:
  - `BENCHMARK_COMPETITIVE_SURPASS_INDEX_KNOWLEDGE_SOURCE_DRIFT_JSON`
- Updated `Build benchmark competitive surpass index (optional)` to pass:
  - dispatch input JSON
  - step output JSON from `benchmark_knowledge_source_drift`
  - env override JSON
- Updated workflow contract tests to assert:
  - dispatch input is present
  - build script includes `--benchmark-knowledge-source-drift`

## Validation
```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path('.github/workflows/evaluation-report.yml').read_text())
print('yaml_ok')
PY
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
git diff --check
```

## Result
- `py_compile`: passed
- `flake8`: passed
- workflow YAML parse: passed
- `pytest`: `3 passed, 1 warning`
- `git diff --check`: passed

## Notes
- This change only wires the input into CI/export generation.
- The standalone competitive surpass exporter logic is already covered by PR `#311`.
