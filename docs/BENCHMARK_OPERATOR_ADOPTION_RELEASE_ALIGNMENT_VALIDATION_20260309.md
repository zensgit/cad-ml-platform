# Benchmark Operator Adoption Release Alignment Validation

## Scope

Extend standalone `benchmark operator adoption` so it no longer treats release
decision and release runbook as a single opaque status. The exporter now
captures whether those two release surfaces align on:

- standalone operator-adoption status
- scorecard operator-adoption status
- operational operator-adoption status
- scorecard operator outcome drift
- operational operator outcome drift

## Files

- `scripts/export_benchmark_operator_adoption.py`
- `tests/unit/test_benchmark_operator_adoption.py`

## Added Output

New payload fields:

- `release_surface_alignment_status`
- `release_surface_alignment_summary`
- `release_surface_alignment`

The nested alignment payload includes:

- `release_decision`
- `release_runbook`
- `mismatches`

## Validation

Executed:

```bash
python3 -m py_compile \
  scripts/export_benchmark_operator_adoption.py \
  tests/unit/test_benchmark_operator_adoption.py

flake8 \
  scripts/export_benchmark_operator_adoption.py \
  tests/unit/test_benchmark_operator_adoption.py \
  --max-line-length=100

pytest -q tests/unit/test_benchmark_operator_adoption.py
```

Expected result:

- alignment is reported as `aligned` when release decision and runbook agree
- markdown output includes a dedicated `Release Surface Alignment` section
- prior operator-adoption guidance behavior remains intact
