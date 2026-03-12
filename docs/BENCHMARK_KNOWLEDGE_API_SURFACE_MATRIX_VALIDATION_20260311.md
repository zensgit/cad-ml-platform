# Benchmark Knowledge API Surface Matrix Validation

## Goal

Add a benchmark control-plane view for public API coverage of core knowledge
domains, with an explicit blocked signal when a domain has knowledge assets but
no stable external API surface.

## Scope

- `src/core/benchmark/knowledge_domain_api_surface_matrix.py`
- `scripts/export_benchmark_knowledge_domain_api_surface_matrix.py`
- downstream benchmark surfaces:
  - artifact bundle
  - companion summary
  - release decision
  - release runbook
- unit validation:
  - `tests/unit/test_benchmark_knowledge_domain_api_surface_matrix.py`

## Design

The component evaluates three benchmark-facing knowledge domains:

- `tolerance`
- `standards`
- `gdt`

For each domain it combines:

- capability status from `knowledge_domain_capability_matrix`
- built-in reference inventory counts
- static route scan of public FastAPI modules under `src/api/v1/`

This keeps the signal stable and deterministic:

- no runtime app bootstrap is required
- route coverage is based on source-controlled decorators
- missing files are treated as explicit public API gaps

## Expected Behavior

- `tolerance` is `ready`
- `standards` is `ready`
- `gdt` is `blocked`
  - because `src/api/v1/gdt.py` is missing
  - even though knowledge capability/reference coverage exists

Downstream benchmark surfaces must expose:

- `knowledge_domain_api_surface_matrix_status`
- per-domain passthrough under `knowledge_domain_api_surface_matrix_domains`
- recommendations in bundle / companion / release surfaces

## Verification

Commands:

```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_domain_api_surface_matrix.py \
  scripts/export_benchmark_knowledge_domain_api_surface_matrix.py \
  tests/unit/test_benchmark_knowledge_domain_api_surface_matrix.py

flake8 \
  src/core/benchmark/knowledge_domain_api_surface_matrix.py \
  scripts/export_benchmark_knowledge_domain_api_surface_matrix.py \
  tests/unit/test_benchmark_knowledge_domain_api_surface_matrix.py \
  --max-line-length=100

pytest -q tests/unit/test_benchmark_knowledge_domain_api_surface_matrix.py
```

Coverage focus:

- blocked public API gap for `gdt`
- recommendation text includes the API surface gap
- artifact bundle passthrough
- companion summary passthrough
- release decision / release runbook review signals

## Outcome

This benchmark layer now distinguishes:

- knowledge exists internally
- knowledge is externally benchmarkable through stable API routes

That closes a control-plane gap for `standards / tolerance / GD&T` and makes
missing public API surfaces visible in release-facing benchmark views.
