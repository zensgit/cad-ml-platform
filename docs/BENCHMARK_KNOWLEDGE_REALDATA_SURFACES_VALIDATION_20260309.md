# Benchmark Knowledge Real-Data Surfaces Validation

## Scope

This delivery promotes `knowledge real-data correlation` from a standalone
benchmark exporter into the main benchmark surfaces:

- companion summary
- artifact bundle
- release decision
- release runbook

The goal is to expose domain-level correlation status for `tolerance`,
`standards`, and `gdt` wherever benchmark operators already review release
readiness.

## Changes

- Added optional `--benchmark-knowledge-realdata-correlation` input to:
  - `scripts/export_benchmark_companion_summary.py`
  - `scripts/export_benchmark_artifact_bundle.py`
  - `scripts/export_benchmark_release_decision.py`
  - `scripts/export_benchmark_release_runbook.py`
- Added surface payload fields:
  - `knowledge_realdata_correlation_status`
  - `knowledge_realdata_correlation_domains`
  - `knowledge_realdata_correlation_priority_domains`
  - `knowledge_realdata_correlation_recommendations`
- Companion and bundle component status maps now expose:
  - `knowledge_realdata_correlation`
- Release decision and release runbook now also surface correlation
  recommendations as review signals when the correlation is not ready.

## Validation

Commands:

```bash
python3 -m py_compile \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_realdata_surfaces.py

flake8 \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_realdata_surfaces.py \
  --max-line-length=100

pytest -q tests/unit/test_benchmark_knowledge_realdata_surfaces.py
```

Results:

- `py_compile`: passed
- `flake8`: passed
- `pytest`: passed

## Notes

- This layer only promotes correlation to standalone benchmark surfaces.
- CI wiring, job summary, and PR comment propagation are intentionally left to
  the next stacked deliveries.
