# Benchmark Operator Adoption Knowledge Drift Validation

## Goal

Teach `benchmark operator adoption` to consume `knowledge_drift` directly so that
operator guidance reflects knowledge regressions instead of only release/runbook status.

## Design

This branch adds a dedicated `knowledge_drift` payload to
`scripts/export_benchmark_operator_adoption.py`.

The exporter now:

- accepts `--benchmark-knowledge-drift`
- reads drift status / summary / recommendations from:
  - explicit knowledge drift JSON
  - release decision surface
  - release runbook surface
- exposes:
  - `knowledge_drift_status`
  - `knowledge_drift_summary`
  - `knowledge_drift`
- downgrades operator readiness when drift is `regressed`
- switches `operator_mode` to `stabilize_knowledge` on regressions
- promotes drift recommendations into the operator action list

## Changed Files

- `scripts/export_benchmark_operator_adoption.py`
- `tests/unit/test_benchmark_operator_adoption.py`

## Validation

Commands run:

```bash
python3 -m py_compile scripts/export_benchmark_operator_adoption.py

flake8 \
  scripts/export_benchmark_operator_adoption.py \
  tests/unit/test_benchmark_operator_adoption.py \
  --max-line-length=100

pytest -q tests/unit/test_benchmark_operator_adoption.py
```

Results:

- `py_compile`: passed
- `flake8`: passed
- `pytest`: `4 passed`

## Outcome

Operator adoption no longer treats knowledge regressions as invisible. When
knowledge coverage regresses:

- adoption falls back from `operator_ready` to `guided_manual`
- operator mode becomes `stabilize_knowledge`
- drift recommendations become top-level operator actions

This gives the release surface a clearer bridge from benchmark regression to
human action.
