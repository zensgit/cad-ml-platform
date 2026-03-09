# Benchmark Operator Adoption Knowledge Outcome Drift Validation

## Scope
- Extend `export_benchmark_operator_adoption.py` to consume `benchmark_knowledge_outcome_drift`
- Surface outcome-drift status, summary, recommendations, and domain changes in operator adoption output
- Treat knowledge outcome regressions as guided-manual stabilization signals

## Changed Files
- `scripts/export_benchmark_operator_adoption.py`
- `tests/unit/test_benchmark_operator_adoption.py`

## Key Changes
- Added CLI input:
  - `--benchmark-knowledge-outcome-drift`
- Added payload fields:
  - `knowledge_outcome_drift_status`
  - `knowledge_outcome_drift_summary`
  - `knowledge_outcome_drift`
- Added artifact tracking:
  - `benchmark_knowledge_outcome_drift`
- Updated operator readiness / mode / action selection to react to outcome regressions

## Validation
```bash
python3 -m py_compile scripts/export_benchmark_operator_adoption.py tests/unit/test_benchmark_operator_adoption.py
flake8 scripts/export_benchmark_operator_adoption.py tests/unit/test_benchmark_operator_adoption.py --max-line-length=100
pytest -q tests/unit/test_benchmark_operator_adoption.py
```

## Results
- `py_compile`: passed
- `flake8`: passed
- `pytest`: `5 passed, 1 warning`

## Notes
- This branch only extends the exporter and its unit contract.
- CI / PR comment wiring for operator-adoption + outcome-drift should stack after this branch.
