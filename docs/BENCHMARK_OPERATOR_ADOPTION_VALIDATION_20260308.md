## Scope

This exporter turns benchmark release artifacts into an operator-facing adoption
summary.

Inputs:

- benchmark release decision
- benchmark release runbook
- active-learning review queue report
- feedback flywheel benchmark

Outputs:

- `adoption_readiness`
- `operator_mode`
- `next_action`
- queue / feedback counters
- recommended actions

## Validation

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

- blocked scenario maps to `clear_blockers`
- ready scenario maps to `freeze_ready`
- review scenario maps to `drive_review`
