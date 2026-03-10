# Benchmark Knowledge Domain Release Surface Alignment Validation

## Scope

This change adds a standalone `knowledge_domain_release_surface_alignment`
benchmark component and propagates it into the benchmark downstream surfaces:

- `benchmark artifact bundle`
- `benchmark companion summary`
- `competitive_surpass_index`

The component compares release-facing `knowledge_domain_control_plane` signals
between:

- `benchmark_release_decision`
- `benchmark_release_runbook`

and produces a stable alignment contract for `standards / tolerance / GD&T`
control-plane readiness.

## Design

### New benchmark component

Implemented in:

- `src/core/benchmark/knowledge_domain_release_surface_alignment.py`

Output contract:

- `status`
- `summary`
- `mismatch_count`
- `mismatches`
- `domain_mismatches`
- `release_blocker_mismatches`
- `release_decision`
- `release_runbook`

Status mapping:

- `aligned`
- `diverged`
- `unavailable`

### Standalone exporter

Implemented in:

- `scripts/export_benchmark_knowledge_domain_release_surface_alignment.py`

CLI inputs:

- `--benchmark-release-decision`
- `--benchmark-release-runbook`

CLI outputs:

- `--output-json`
- `--output-md`

### Downstream surface propagation

Updated:

- `scripts/export_benchmark_artifact_bundle.py`
- `scripts/export_benchmark_companion_summary.py`
- `src/core/benchmark/competitive_surpass_index.py`
- `scripts/export_benchmark_competitive_surpass_index.py`

New downstream fields:

- `knowledge_domain_release_surface_alignment`
- `knowledge_domain_release_surface_alignment_status`
- `knowledge_domain_release_surface_alignment_summary`
- `knowledge_domain_release_surface_alignment_mismatches`

Artifact bundle additionally exposes:

- `knowledge_domain_release_surface_alignment_domain_mismatches`
- `knowledge_domain_release_surface_alignment_release_blocker_mismatches`

`competitive_surpass_index` now includes this component in the
`release_alignment` pillar alongside operator adoption release alignment.

## Validation

Commands:

```bash
pytest -q \
  tests/unit/test_benchmark_knowledge_domain_release_surface_alignment.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_competitive_surpass_index.py

python3 -m py_compile \
  src/core/benchmark/knowledge_domain_release_surface_alignment.py \
  scripts/export_benchmark_knowledge_domain_release_surface_alignment.py \
  scripts/export_benchmark_competitive_surpass_index.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py

flake8 \
  src/core/benchmark/knowledge_domain_release_surface_alignment.py \
  scripts/export_benchmark_knowledge_domain_release_surface_alignment.py \
  scripts/export_benchmark_competitive_surpass_index.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  tests/unit/test_benchmark_knowledge_domain_release_surface_alignment.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_competitive_surpass_index.py \
  --max-line-length=100

git diff --check
```

Results:

- `pytest`: `27 passed, 1 warning`
- `py_compile`: passed
- `flake8`: passed
- `git diff --check`: passed

## Notes

- The new release-alignment pillar is intentionally stricter than the previous
  operator-only release view.
- A release alignment is now only fully ready when both:
  - operator adoption release surfaces are aligned
  - knowledge-domain control-plane release surfaces are aligned
