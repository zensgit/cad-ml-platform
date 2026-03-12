# Benchmark Knowledge Subdomain Surface Matrix Validation

## Goal

Add a finer-grained benchmark control-plane artifact for knowledge subdomains so
`tolerance`, `standards`, `design_standards`, and `gdt` can be evaluated below
the domain level and surfaced consistently through bundle, companion, release
decision, and release runbook outputs.

## Key Changes

- Added component:
  - `src/core/benchmark/knowledge_subdomain_surface_matrix.py`
- Added exporter:
  - `scripts/export_benchmark_knowledge_subdomain_surface_matrix.py`
- Wired downstream surfaces:
  - `scripts/export_benchmark_artifact_bundle.py`
  - `scripts/export_benchmark_companion_summary.py`
  - `scripts/export_benchmark_release_decision.py`
  - `scripts/export_benchmark_release_runbook.py`
- Added tests:
  - `tests/unit/test_benchmark_knowledge_subdomain_surface_matrix.py`
  - `tests/unit/test_benchmark_knowledge_subdomain_surface_matrix_surfaces.py`

## Validation

Commands:

```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_subdomain_surface_matrix.py \
  scripts/export_benchmark_knowledge_subdomain_surface_matrix.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_subdomain_surface_matrix.py \
  tests/unit/test_benchmark_knowledge_subdomain_surface_matrix_surfaces.py

flake8 \
  src/core/benchmark/knowledge_subdomain_surface_matrix.py \
  scripts/export_benchmark_knowledge_subdomain_surface_matrix.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_subdomain_surface_matrix.py \
  tests/unit/test_benchmark_knowledge_subdomain_surface_matrix_surfaces.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_knowledge_subdomain_surface_matrix.py \
  tests/unit/test_benchmark_knowledge_subdomain_surface_matrix_surfaces.py
```

Results:

- `py_compile`: passed
- `flake8`: passed
- `pytest`: `5 passed, 1 warning`

## Outcome

The new benchmark artifact now exposes subdomain-level public API and reference
coverage gaps and propagates that information into all major benchmark control
surfaces without regressing existing release decision or release runbook logic.
