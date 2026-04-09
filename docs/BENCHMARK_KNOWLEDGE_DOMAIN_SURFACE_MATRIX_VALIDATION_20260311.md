# Benchmark Knowledge Domain Surface Matrix Validation

## Goal

Add a finer-grained benchmark control-plane over public knowledge sub-capabilities so
`tolerance`, `standards`, and `GD&T` can be measured below the domain summary level.

## Scope

- Added `knowledge_domain_surface_matrix` benchmark component
- Added standalone exporter:
  - `scripts/export_benchmark_knowledge_domain_surface_matrix.py`
- Added downstream surfaces passthrough:
  - `scripts/export_benchmark_artifact_bundle.py`
  - `scripts/export_benchmark_companion_summary.py`
  - `scripts/export_benchmark_release_decision.py`
  - `scripts/export_benchmark_release_runbook.py`
- Added markdown rendering and recommendations
- Added unit coverage for:
  - route-gap detection and markdown output
  - bundle / companion passthrough
  - release decision / runbook review signals

## Validation

```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_domain_surface_matrix.py \
  scripts/export_benchmark_knowledge_domain_surface_matrix.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_domain_surface_matrix.py \
  tests/unit/test_benchmark_knowledge_domain_surface_matrix_surfaces.py

python3 -m flake8 \
  src/core/benchmark/knowledge_domain_surface_matrix.py \
  scripts/export_benchmark_knowledge_domain_surface_matrix.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_domain_surface_matrix.py \
  tests/unit/test_benchmark_knowledge_domain_surface_matrix_surfaces.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_knowledge_domain_surface_matrix.py \
  tests/unit/test_benchmark_knowledge_domain_surface_matrix_surfaces.py \
  tests/unit/test_benchmark_release_runbook.py \
  -k 'surface_matrix or render_markdown_and_cli_outputs or freezes_when_ready'
```

## Expected result

- Exporter builds successfully
- Matrix reports missing public GD&T benchmark routes
- Tolerance public surfaces are detected as present
- Markdown rendering includes domain rows and recommendations
- Bundle / companion surfaces preserve matrix status and domain details
- Release decision / runbook surfaces expose surface-matrix review signals without
  regressing release-runbook priority handling
