# Benchmark Release Scorecard Operator Adoption Validation

## Scope

- Extend `benchmark release decision` with scorecard and operational operator-adoption
  passthrough.
- Extend `benchmark release runbook` with the same downstream surfaces.
- Keep existing standalone operator-adoption release semantics unchanged.

## Design

- `release decision` now exports:
  - `scorecard_operator_adoption`
  - `operational_operator_adoption`
- `release runbook` now exports:
  - `scorecard_operator_adoption`
  - `operational_operator_adoption`
- Both surfaces include:
  - `status`
  - `operator_mode` where available
  - `knowledge_outcome_drift_status`
  - `knowledge_outcome_drift_summary`
- Markdown renderers add:
  - `## Scorecard Operator Adoption`
  - `## Operational Operator Adoption`

## Files

- `scripts/export_benchmark_release_decision.py`
- `scripts/export_benchmark_release_runbook.py`
- `tests/unit/test_benchmark_release_decision.py`
- `tests/unit/test_benchmark_release_runbook.py`

## Validation

```bash
python3 -m py_compile \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py

flake8 \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py
```

## Result

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `12 passed`

## Notes

- `release runbook` did not previously accept direct `benchmark_scorecard` or
  `benchmark_operational_summary` inputs. This change adds both inputs so the
  exporter can surface the same operator-adoption evidence already present in
  bundle/companion layers.
