# Benchmark Knowledge Drift CI Validation

## Scope
- Added standalone `benchmark knowledge drift` wiring to [evaluation-report.yml](/private/tmp/cad-ml-platform-benchmark-knowledge-drift-ci-v2/.github/workflows/evaluation-report.yml).
- Added artifact upload, job summary, and PR comment exposure for `knowledge drift`.

## Implemented
- New workflow env/config:
  - `BENCHMARK_KNOWLEDGE_DRIFT_ENABLE`
  - `BENCHMARK_KNOWLEDGE_DRIFT_TITLE`
  - `BENCHMARK_KNOWLEDGE_DRIFT_CURRENT_SUMMARY_JSON`
  - `BENCHMARK_KNOWLEDGE_DRIFT_PREVIOUS_SUMMARY_JSON`
  - `BENCHMARK_KNOWLEDGE_DRIFT_OUTPUT_JSON`
  - `BENCHMARK_KNOWLEDGE_DRIFT_OUTPUT_MD`
- New workflow step:
  - `Build benchmark knowledge drift (optional)`
- New artifact upload:
  - `Upload benchmark knowledge drift`
- New summary lines:
  - drift status
  - current/previous status
  - reference item delta
  - regressions/improvements
  - resolved/new focus areas
  - recommendations
- New PR comment / signal-light wiring:
  - `Benchmark Knowledge Drift`
  - `Benchmark Knowledge Drift Recommendations`

## Validation
```bash
python3 - <<'PY'
import yaml, pathlib
path = pathlib.Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text())
print('yaml_ok')
PY

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Result
- YAML parse: passed
- `flake8`: passed
- `pytest`: `3 passed`

## Notes
- This branch only wires the standalone `knowledge drift` artifact into CI.
- Bundle/companion and release-surface consumption are intentionally left to sibling branches to avoid workflow-heavy conflicts.
