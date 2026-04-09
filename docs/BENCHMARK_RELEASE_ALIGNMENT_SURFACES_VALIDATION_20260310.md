# Benchmark Release Alignment Surfaces Validation

## Goal

Expose operator-adoption `release_surface_alignment` in:

- `benchmark release decision`
- `benchmark release runbook`

This makes release-facing benchmark surfaces carry the same alignment signal
already present in the standalone operator-adoption exporter.

## Changes

- Added release-surface alignment extraction helpers to:
  - `scripts/export_benchmark_release_decision.py`
  - `scripts/export_benchmark_release_runbook.py`
- Added payload field:
  - `operator_adoption_release_surface_alignment`
- Added markdown section:
  - `## Operator Adoption Release Surface Alignment`
- Extended unit tests for:
  - fallback `unknown`
  - explicit aligned state
  - explicit mismatched state

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
- `pytest`: `14 passed`

## Outcome

Release-facing benchmark surfaces now carry operator-adoption alignment status,
summary, and mismatch detail. The next step is to wire these fields into
workflow outputs, job summary, and PR comments.
