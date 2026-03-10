# Benchmark Operator Adoption Release Surfaces Validation

## Goal

Expose `release_surface_alignment` from `benchmark_operator_adoption` in:

- `benchmark artifact bundle`
- `benchmark companion summary`

This closes the gap between the standalone exporter and downstream benchmark
surfaces.

## Changes

- Added release-surface alignment extraction in:
  - `scripts/export_benchmark_artifact_bundle.py`
  - `scripts/export_benchmark_companion_summary.py`
- Added downstream payload fields:
  - `operator_adoption_release_surface_alignment.status`
  - `operator_adoption_release_surface_alignment.summary`
  - `operator_adoption_release_surface_alignment.mismatches`
- Added markdown sections:
  - `## Operator Adoption Release Surface Alignment`
- Extended unit fixtures for:
  - aligned release surfaces
  - mismatched release surfaces
  - missing release-surface alignment fallback

## Validation

```bash
python3 -m py_compile \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py

flake8 \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py
```

## Result

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `12 passed`

## Outcome

`release_surface_alignment` is now available in bundle/companion surfaces, so
the next CI layer can summarize and comment on alignment instead of relying only
on the standalone operator-adoption exporter output.
