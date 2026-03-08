# Benchmark Scorecard CI Validation 2026-03-08

## Scope

This change wires the benchmark scorecard generator into
`.github/workflows/evaluation-report.yml` as an optional artifact-producing step.

The goal is to move benchmark reporting from "manual script only" to
"CI-discoverable artifact and summary signal".

## Added Workflow Surface

- `BENCHMARK_SCORECARD_ENABLE`
- benchmark scorecard summary input env vars
- `Generate benchmark scorecard (optional)`
- `Upload benchmark scorecard`
- job summary lines for benchmark status
- PR comment lines for benchmark status and recommendations

## Validation

```bash
python3 -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('.github/workflows/evaluation-report.yml').read_text())"

python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Expected Result

- workflow YAML parses successfully
- workflow regression tests continue to cover Graph2D review pack / train sweep
- workflow regression tests additionally verify benchmark scorecard env, steps,
  artifacts, summary lines, and PR comment contract
