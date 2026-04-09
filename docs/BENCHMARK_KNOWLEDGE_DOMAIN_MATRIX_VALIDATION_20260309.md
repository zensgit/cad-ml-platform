# Benchmark Knowledge Domain Matrix Validation

## Goal

Add a benchmark-level `knowledge_domain_matrix` layer that aggregates three
existing benchmark signals:

- `knowledge_readiness`
- `knowledge_application`
- `knowledge_realdata_correlation`

The new layer provides a domain-oriented view for `tolerance`, `standards`,
and `gdt`, then feeds that view into the existing benchmark surfaces:

- companion summary
- artifact bundle
- release decision
- release runbook

## Design

### New benchmark component

Added [knowledge_domain_matrix.py](../src/core/benchmark/knowledge_domain_matrix.py)
with:

- `build_knowledge_domain_matrix_status(...)`
- `knowledge_domain_matrix_recommendations(...)`
- `render_knowledge_domain_matrix_markdown(...)`

The aggregator computes:

- per-domain `status`
- per-domain `priority`
- `focus_components`
- `missing_metrics`
- application signal counts
- real-data ready / partial / blocked / missing components
- a top-level focus list and recommended actions

### New standalone exporter

Added
[export_benchmark_knowledge_domain_matrix.py](../scripts/export_benchmark_knowledge_domain_matrix.py)
to emit JSON + Markdown outputs from the three upstream benchmark artifacts.

### Surface integration

Integrated the new artifact into:

- [export_benchmark_companion_summary.py](../scripts/export_benchmark_companion_summary.py)
- [export_benchmark_artifact_bundle.py](../scripts/export_benchmark_artifact_bundle.py)
- [export_benchmark_release_decision.py](../scripts/export_benchmark_release_decision.py)
- [export_benchmark_release_runbook.py](../scripts/export_benchmark_release_runbook.py)

The new surface fields are:

- `knowledge_domain_matrix_status`
- `knowledge_domain_matrix`
- `knowledge_domain_matrix_domains`
- `knowledge_domain_matrix_priority_domains`
- `knowledge_domain_matrix_recommendations`

### Test coverage

Added
[test_benchmark_knowledge_domain_matrix.py](../tests/unit/test_benchmark_knowledge_domain_matrix.py)
and extended surface tests to validate:

- ready / partial / blocked aggregation
- CLI exporter outputs
- markdown rendering
- bundle / companion / release decision / release runbook passthrough

## Validation

Commands run in isolated worktree
`/private/tmp/cad-ml-platform-knowledge-realdata-correlation-20260309`.

```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_domain_matrix.py \
  scripts/export_benchmark_knowledge_domain_matrix.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_domain_matrix.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py

flake8 \
  src/core/benchmark/knowledge_domain_matrix.py \
  scripts/export_benchmark_knowledge_domain_matrix.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_domain_matrix.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_knowledge_domain_matrix.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py
```

Results:

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `22 passed`

## Limitations

- This change only adds exporter + surface integration. CI wiring and PR comment
  exposure are intentionally split into stacked follow-up branches.
- Domain coverage is currently fixed to `tolerance`, `standards`, and `gdt`.
- Real-data depth still depends on upstream benchmark artifact availability.
