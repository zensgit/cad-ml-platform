# Benchmark Knowledge Subdomain Surface Matrix CI Validation

## Goal

Wire `benchmark_knowledge_subdomain_surface_matrix` into
`.github/workflows/evaluation-report.yml` so the artifact can be built,
uploaded, summarized, and propagated into downstream benchmark control-plane
surfaces.

## Key Changes

- Added workflow dispatch toggle:
  - `benchmark_knowledge_subdomain_surface_matrix_enable`
- Added env defaults:
  - `BENCHMARK_KNOWLEDGE_SUBDOMAIN_SURFACE_MATRIX_ENABLE`
  - `BENCHMARK_KNOWLEDGE_SUBDOMAIN_SURFACE_MATRIX_TITLE`
  - `BENCHMARK_KNOWLEDGE_SUBDOMAIN_SURFACE_MATRIX_OUTPUT_JSON`
  - `BENCHMARK_KNOWLEDGE_SUBDOMAIN_SURFACE_MATRIX_OUTPUT_MD`
- Added workflow step:
  - `Build benchmark knowledge subdomain surface matrix (optional)`
- Added artifact upload step
- Added summary lines and downstream passthrough into:
  - artifact bundle
  - companion summary
  - release decision
  - release runbook
- Added workflow contract coverage in:
  - `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation

Commands:

```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Results:

- `py_compile`: passed
- `flake8`: passed
- `pytest`: `13 passed, 1 warning`

## Outcome

The workflow now understands `knowledge_subdomain_surface_matrix` as a first
class benchmark artifact and can carry it through CI/build/reporting surfaces in
the same way as other benchmark control-plane components.
