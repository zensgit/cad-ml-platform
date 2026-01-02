# Prometheus Rules Validation (2026-01-01)

## Scope

- Validate Prometheus recording rules with the local validator and promtool.

## Command

- `make prom-validate`

## Results

- OK: validation passed; promtool check rules succeeded (25 rules).

## Notes

- Validator reports two naming warnings (`top_rejection_reason_rate`, `memory_exhaustion_rate`).
- Updated Makefile promtool invocation to use `--entrypoint promtool` to avoid "unexpected promtool" errors.
