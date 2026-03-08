# Benchmark Release Decision Operator Adoption Validation 2026-03-08

## Goal

Extend `scripts/export_benchmark_release_decision.py` so release decisions can
see operator-adoption readiness without letting that lower-priority signal
override stronger release, companion, bundle, or engineering evidence.

## Scope

Changed files:

- `scripts/export_benchmark_release_decision.py`
- `tests/unit/test_benchmark_release_decision.py`
- `docs/BENCHMARK_RELEASE_DECISION_OPERATOR_ADOPTION_VALIDATION_20260308.md`

## Changes

### New CLI input

- `--benchmark-operator-adoption`

The script now loads the `benchmark_operator_adoption` JSON summary and threads
it through release-decision generation.

### New release-decision visibility

- `component_statuses.operator_adoption`
- `artifacts.benchmark_operator_adoption`

This makes operator-adoption readiness visible in both JSON and Markdown
outputs.

### Priority rule

Operator adoption is treated as a low-priority fallback only:

- `blocking_signals` fall back to `benchmark_operator_adoption.blocking_signals`
  only when companion, bundle, and operational blockers are empty
- `review_signals` fall back to operator-adoption recommendations only when
  release/companion/bundle/scorecard/engineering review signals are empty
- automatic critical/review status scanning excludes `operator_adoption`, so
  its status is informational unless it enters through those fallback lists

This preserves the existing higher-priority decision chain.

## Test Coverage

Added or updated tests for:

- exposing `operator_adoption` in `component_statuses`
- exposing `benchmark_operator_adoption` in artifact paths
- proving operator adoption does not override companion blockers or higher
  priority review signals
- fallback blocker behavior when higher-priority blockers are absent
- fallback recommendation behavior when higher-priority review signals are
  absent
- CLI wiring and Markdown rendering

## Validation

```bash
python3 -m py_compile scripts/export_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_decision.py

flake8 scripts/export_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_decision.py \
  --max-line-length=100

pytest -q tests/unit/test_benchmark_release_decision.py
```

## Validation Result

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `5 passed`
- warning note: one existing `PendingDeprecationWarning` from `starlette`
  importing `multipart`; no code changes were made for that external warning
