# Benchmark Release Operator Outcome Drift Validation

## Goal
- Expose `operator_adoption_knowledge_outcome_drift` on benchmark release surfaces.
- Ensure both release decision and release runbook can consume operator-adoption outcome-drift guidance without relying on workflow-only glue.

## Scope
- `scripts/export_benchmark_release_decision.py`
- `scripts/export_benchmark_release_runbook.py`
- `tests/unit/test_benchmark_release_decision.py`
- `tests/unit/test_benchmark_release_runbook.py`

## Changes
- Added `operator_adoption_knowledge_outcome_drift` payload extraction to release decision.
- Added operator-adoption outcome-drift recommendation escalation into release decision review signals.
- Added release decision markdown fields:
  - `operator_adoption_knowledge_outcome_drift`
  - `operator_adoption_knowledge_outcome_drift_summary`
- Extended release runbook operator-adoption payload with:
  - `knowledge_outcome_drift_status`
  - `knowledge_outcome_drift_summary`
  - `knowledge_outcome_drift_recommendations`
- Extended release runbook markdown `## Operator Adoption` section with outcome-drift fields and recommendation rows.

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

git diff --check

pytest -q \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py
```

## Result
- `py_compile`: passed
- `flake8`: passed
- `git diff --check`: passed
- `pytest`: `10 passed, 1 warning`

## Outcome
- Release decision now treats operator-adoption outcome drift as a first-class review signal.
- Release runbook now renders operator-adoption outcome drift explicitly, including recommended follow-up.
- Downstream CI wiring can consume stable fields instead of inferring them from free-form operator adoption summaries.
