# Benchmark Knowledge Outcome Drift Validation

## Goal

Add a standalone `knowledge outcome drift` exporter so benchmark knowledge
outcome correlation can be compared against a previous baseline before wiring it
into higher-level CI and release surfaces.

## Implementation

Primary files:

- `src/core/benchmark/knowledge_outcome_drift.py`
- `src/core/benchmark/__init__.py`
- `scripts/export_benchmark_knowledge_outcome_drift.py`
- `tests/unit/test_benchmark_knowledge_outcome_drift.py`

## Added Coverage

The new helper/exporter provides:

- overall drift status:
  - `baseline_missing`
  - `stable`
  - `improved`
  - `regressed`
  - `mixed`
- domain-level change tracking
- focus-area and priority-domain resolution/newness
- best-surface score deltas
- markdown rendering and JSON CLI export

## Validation Commands

```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_outcome_drift.py \
  scripts/export_benchmark_knowledge_outcome_drift.py \
  tests/unit/test_benchmark_knowledge_outcome_drift.py

flake8 \
  src/core/benchmark/knowledge_outcome_drift.py \
  scripts/export_benchmark_knowledge_outcome_drift.py \
  tests/unit/test_benchmark_knowledge_outcome_drift.py \
  --max-line-length=100

pytest -q tests/unit/test_benchmark_knowledge_outcome_drift.py
```

## Validation Result

- `py_compile`: pass
- `flake8`: pass
- `pytest`: pass

## Outcome

`knowledge outcome correlation` now has its own drift/baseline exporter, making
it possible to measure whether the benchmark knowledge-to-realdata alignment is
improving or regressing before wiring it into CI, companion, bundle, release,
and runbook surfaces.
