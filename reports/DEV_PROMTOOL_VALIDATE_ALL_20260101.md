# Prometheus Rules Validation (Promtool All) (2026-01-01)

## Scope

- Validate all Prometheus rule files with promtool (alerting + recording).

## Command

- `make promtool-validate-all`

## Results

- OK: all rule files validated successfully (alerting: 49 rules; recording: 28 + 25 rules).

## Notes

- Updated Docker fallback in `scripts/validate_prometheus.sh` to use `--entrypoint promtool`.
