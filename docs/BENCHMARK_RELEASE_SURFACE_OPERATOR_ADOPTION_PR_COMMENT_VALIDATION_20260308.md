# Benchmark Release Surface Operator Adoption PR Comment Validation 2026-03-08

## Goal

Expose release-surface operator-adoption status directly in the benchmark PR
comment so release decision and release runbook lines carry the same
operator-adoption visibility that now exists in CI step outputs.

## Scope

Changed files:

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
- `docs/BENCHMARK_RELEASE_SURFACE_OPERATOR_ADOPTION_PR_COMMENT_VALIDATION_20260308.md`

## Changes

- Added PR-comment JS bindings:
  - `benchmarkReleaseOperatorAdoptionStatus`
  - `benchmarkReleaseRunbookOperatorAdoptionStatus`
- Extended release decision status line with:
  - `operator_adoption=${benchmarkReleaseOperatorAdoptionStatus}`
- Extended release runbook status line with:
  - `operator_adoption=${benchmarkReleaseRunbookOperatorAdoptionStatus}`

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
