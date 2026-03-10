# Benchmark Knowledge Source Coverage Validation

## Scope

This delivery adds a new benchmark component, `knowledge_source_coverage`, to
measure whether the built-in engineering knowledge bases are materially covered
by source tables and reference standards, and to expose that status across the
existing benchmark surfaces.

Integrated surfaces in this layer:

- standalone exporter
- artifact bundle
- companion summary
- release decision
- release runbook
- competitive surpass index

## Design

### New benchmark component

Files:

- `src/core/benchmark/knowledge_source_coverage.py`
- `scripts/export_benchmark_knowledge_source_coverage.py`

The component builds a deterministic snapshot from built-in knowledge modules
and groups them into:

- core source groups:
  - `tolerance`
  - `standards`
  - `design_standards`
  - `gdt`
- expansion candidates:
  - `machining`
  - `welding`
  - `surface_treatment`
  - `heat_treatment`

Each group emits:

- `status`
- `priority`
- `source_table_count`
- `ready_source_table_count`
- `missing_source_table_count`
- `source_item_count`
- `reference_standard_count`
- `missing_source_tables`

### Status model

Core groups require:

- non-zero source table count
- all source tables non-zero
- non-zero source items
- non-zero reference standards

Expansion groups are intentionally allowed to be `ready` without reference
standards when they provide meaningful built-in manufacturing tables. This is
implemented through `allow_reference_free=True`.

### Downstream propagation

This layer propagates `knowledge_source_coverage` into:

- `scripts/export_benchmark_artifact_bundle.py`
- `scripts/export_benchmark_companion_summary.py`
- `scripts/export_benchmark_release_decision.py`
- `scripts/export_benchmark_release_runbook.py`
- `src/core/benchmark/competitive_surpass_index.py`

New exposed fields include:

- `knowledge_source_coverage_status`
- `knowledge_source_coverage`
- `knowledge_source_coverage_domains`
- `knowledge_source_coverage_expansion_candidates`
- `knowledge_source_coverage_recommendations`

The competitive surpass knowledge pillar also now treats
`knowledge_source_coverage_ready` and `knowledge_source_coverage_partial` as
first-class tier inputs.

## Key Fixes During Implementation

- Corrected `design_standards` table key from `standard_filletts` to
  `standard_fillets`
- Added the missing artifact row/path handling for
  `benchmark_knowledge_source_coverage` in companion summary
- Updated tests and release/runbook expectations so the new artifact is
  explicitly present when the surface is expected to be freeze-ready

## Validation

### Syntax

```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_source_coverage.py \
  scripts/export_benchmark_knowledge_source_coverage.py \
  src/core/benchmark/competitive_surpass_index.py \
  scripts/export_benchmark_competitive_surpass_index.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_source_coverage.py \
  tests/unit/test_benchmark_competitive_surpass_index.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py
```

Result: pass

### Lint

```bash
flake8 \
  src/core/benchmark/knowledge_source_coverage.py \
  scripts/export_benchmark_knowledge_source_coverage.py \
  src/core/benchmark/competitive_surpass_index.py \
  scripts/export_benchmark_competitive_surpass_index.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_source_coverage.py \
  tests/unit/test_benchmark_competitive_surpass_index.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py \
  --max-line-length=100
```

Result: pass

### Targeted tests

```bash
pytest -q \
  tests/unit/test_benchmark_knowledge_source_coverage.py \
  tests/unit/test_benchmark_competitive_surpass_index.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py
```

Result:

- `34 passed`
- `1 warning` (`python_multipart` pending deprecation from dependency stack)

## Outcome

The benchmark control-plane now has a concrete source-coverage signal for core
knowledge assets, plus explicit promotion signals for manufacturing expansion
areas. This closes a previous blind spot where benchmark knowledge readiness
could be reported without showing whether the underlying built-in source tables
were actually present and populated.
