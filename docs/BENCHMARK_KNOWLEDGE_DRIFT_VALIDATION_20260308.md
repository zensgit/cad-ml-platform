# Benchmark Knowledge Drift Validation

## Goal

Add a benchmark signal that compares current knowledge readiness against a
previous baseline so release surfaces can later answer whether knowledge
coverage improved, stayed stable, or regressed.

## Design

- Add `knowledge_drift` helper and exporter.
- Compare:
  - overall status
  - component status transitions
  - total reference item delta
  - focus-area additions/resolutions
- Emit drift recommendations for:
  - missing baseline
  - improved
  - regressed
  - mixed

## Files

- `src/core/benchmark/knowledge_drift.py`
- `src/core/benchmark/__init__.py`
- `scripts/export_benchmark_knowledge_drift.py`
- `tests/unit/test_benchmark_knowledge_drift.py`

## Validation

```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_drift.py \
  scripts/export_benchmark_knowledge_drift.py

flake8 \
  src/core/benchmark/knowledge_drift.py \
  src/core/benchmark/__init__.py \
  scripts/export_benchmark_knowledge_drift.py \
  tests/unit/test_benchmark_knowledge_drift.py \
  --max-line-length=100

pytest -q tests/unit/test_benchmark_knowledge_drift.py
```

Expected result:

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `3 passed`
