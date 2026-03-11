# Benchmark Knowledge Domain Surface Matrix Validation

## Goal

Add a finer-grained benchmark control-plane over public knowledge sub-capabilities so
`tolerance`, `standards`, and `GD&T` can be measured below the domain summary level.

## Scope

- Added `knowledge_domain_surface_matrix` benchmark component
- Added standalone exporter:
  - `scripts/export_benchmark_knowledge_domain_surface_matrix.py`
- Added markdown rendering and recommendations
- Added unit coverage for route-gap detection and markdown output

## Validation

```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_domain_surface_matrix.py \
  scripts/export_benchmark_knowledge_domain_surface_matrix.py \
  tests/unit/test_benchmark_knowledge_domain_surface_matrix.py

flake8 \
  src/core/benchmark/knowledge_domain_surface_matrix.py \
  scripts/export_benchmark_knowledge_domain_surface_matrix.py \
  tests/unit/test_benchmark_knowledge_domain_surface_matrix.py \
  --max-line-length=100

pytest -q tests/unit/test_benchmark_knowledge_domain_surface_matrix.py
```

## Expected result

- Exporter builds successfully
- Matrix reports missing public GD&T benchmark routes
- Tolerance public surfaces are detected as present
- Markdown rendering includes domain rows and recommendations
