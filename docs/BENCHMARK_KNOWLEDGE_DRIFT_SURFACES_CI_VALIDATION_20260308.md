# Benchmark Knowledge Drift Surfaces CI Validation

## Goal

Complete the `knowledge_drift` delivery chain beyond the standalone exporter so that
GitHub CI and PR presentation reflect drift status across:

- benchmark artifact bundle
- benchmark companion summary
- benchmark release decision
- benchmark release runbook

## Design

Two gaps were closed together on this branch:

1. `evaluation-report.yml` did not pass `benchmark_knowledge_drift` into the
   downstream bundle / companion / release exporters, even though the exporters
   supported the additional surface.
2. Job summary and PR comment output only displayed the standalone
   `Benchmark Knowledge Drift` artifact, not the downstream surfaces that consume it.

This branch adds:

- workflow dispatch inputs and env vars for:
  - `benchmark_artifact_bundle_knowledge_drift_json`
  - `benchmark_companion_summary_knowledge_drift_json`
  - `benchmark_release_decision_knowledge_drift_json`
  - `benchmark_release_runbook_knowledge_drift_json`
- `add_if_exists --benchmark-knowledge-drift ...` wiring in all four build steps
- parsed outputs for downstream drift status / summary / recommendations
- job summary lines for bundle / companion / release decision / runbook drift
- PR comment rows and status strings that include downstream drift state

## Changed Files

- `.github/workflows/evaluation-report.yml`
- `scripts/export_benchmark_release_decision.py`
- `scripts/export_benchmark_release_runbook.py`
- `tests/unit/test_benchmark_release_decision.py`
- `tests/unit/test_benchmark_release_runbook.py`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation

Commands run:

```bash
python3 - <<'PY'
import yaml, pathlib
path = pathlib.Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text())
print('yaml_ok')
PY

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

python3 -m py_compile \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

pytest -q \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Results:

- workflow YAML parse: passed
- `flake8`: passed
- `py_compile`: passed
- workflow regression: `3 passed`
- combined benchmark surface regression: `21 passed`

## Outcome

`knowledge_drift` is now wired across the benchmark presentation chain instead of
stopping at the standalone artifact. GitHub summary / PR comment can surface:

- downstream drift status
- downstream drift summary
- drift recommendations
- drift component changes

This makes benchmark drift actionable from the same CI surface already used for
scorecard, companion, bundle, release decision, and runbook review.
