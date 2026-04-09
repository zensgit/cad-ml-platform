# Benchmark Knowledge Readiness Validation

## Goal

Promote built-in tolerance, standards, design-standards, and GD&T coverage into
the benchmark stack so release surfaces can reason about knowledge readiness,
not only model or workflow readiness.

## Delivered

### 1. Reusable benchmark helper

- Added [src/core/benchmark/knowledge_readiness.py](../src/core/benchmark/knowledge_readiness.py)
- New reusable outputs:
  - `knowledge_readiness.status`
  - `knowledge_readiness.component_statuses`
  - `knowledge_readiness.coverage_counts`
  - `knowledge_readiness.recommendations`

Built-in knowledge sources now contribute deterministic readiness signals for:

- `tolerance`
- `standards`
- `design_standards`
- `gdt`

### 2. Standalone exporter

- Added [scripts/export_benchmark_knowledge_readiness.py](../scripts/export_benchmark_knowledge_readiness.py)
- Supports:
  - live built-in knowledge snapshot
  - optional `--knowledge-snapshot` override
- Outputs:
  - JSON summary
  - Markdown summary

### 3. Benchmark surface integration

Updated these benchmark exporters to accept and surface
`benchmark_knowledge_readiness`:

- [scripts/generate_benchmark_scorecard.py](../scripts/generate_benchmark_scorecard.py)
- [scripts/export_benchmark_companion_summary.py](../scripts/export_benchmark_companion_summary.py)
- [scripts/export_benchmark_artifact_bundle.py](../scripts/export_benchmark_artifact_bundle.py)
- [scripts/export_benchmark_release_decision.py](../scripts/export_benchmark_release_decision.py)
- [scripts/export_benchmark_release_runbook.py](../scripts/export_benchmark_release_runbook.py)

Delivered behavior:

- scorecard gains `knowledge_readiness` component
- companion summary tracks `knowledge_readiness` in `component_statuses`
- artifact bundle includes `benchmark_knowledge_readiness`
- release decision adds knowledge review signals when readiness is partial/missing
- release runbook carries `knowledge_status` and knowledge-driven review signals

## Validation

Commands:

```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_readiness.py \
  scripts/export_benchmark_knowledge_readiness.py \
  scripts/generate_benchmark_scorecard.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_readiness.py \
  tests/unit/test_generate_benchmark_scorecard.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py

flake8 \
  src/core/benchmark/knowledge_readiness.py \
  scripts/export_benchmark_knowledge_readiness.py \
  scripts/generate_benchmark_scorecard.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_readiness.py \
  tests/unit/test_generate_benchmark_scorecard.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_knowledge_readiness.py \
  tests/unit/test_generate_benchmark_scorecard.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py
```

Result:

- `25 passed`
- `py_compile` passed
- `flake8` passed

## Notes

- This branch intentionally stops at reusable exporter/surface integration.
- `evaluation-report.yml`, artifact upload, job summary, and PR comment wiring
  are handled in a stacked CI branch so the base contract remains small and easy
  to validate.
