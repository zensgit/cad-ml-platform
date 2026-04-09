# Benchmark Release Surface Operator Adoption CI Validation 2026-03-08

## Goal

Wire `benchmark_operator_adoption` into the benchmark release decision and
release runbook workflow steps so the release surfaces consume operator
adoption JSON in CI and expose stable step outputs for summary/reporting.

## Scope

Changed files:

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
- `docs/BENCHMARK_RELEASE_SURFACE_OPERATOR_ADOPTION_CI_VALIDATION_20260308.md`

## Changes

### Workflow inputs and env

Added dispatch/env support for:

- `benchmark_release_decision_operator_adoption_json`
- `benchmark_release_runbook_operator_adoption_json`
- `BENCHMARK_RELEASE_DECISION_OPERATOR_ADOPTION_JSON`
- `BENCHMARK_RELEASE_RUNBOOK_OPERATOR_ADOPTION_JSON`

### Release decision wiring

The `Build benchmark release decision (optional)` step now accepts operator
adoption JSON from:

- explicit workflow input
- the `benchmark_operator_adoption` step output JSON
- environment fallback

New release decision step output:

- `operator_adoption_status`

### Release runbook wiring

The `Build benchmark release runbook (optional)` step now accepts operator
adoption JSON from:

- explicit workflow input
- the `benchmark_operator_adoption` step output JSON
- environment fallback

New release runbook step output:

- `operator_adoption_status`

### Job summary

The summary now includes:

- `Benchmark release operator adoption status`
- `Benchmark release runbook operator adoption status`

## Validation

```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Validation Result

- `py_compile`: pass
- `flake8`: pass
- `pytest`: pass
- workflow YAML parse: pass
