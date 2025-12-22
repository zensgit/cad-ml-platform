# Prometheus Rules Validation Report

## Scope
- Validate Prometheus rule files using `scripts/validate_prom_rules.py` with static checks.

## Test Run
- Command: `.venv/bin/python scripts/validate_prom_rules.py --skip-promtool --json`
- Result: `validation_passed=true`, `rules=25`, `errors=0`

## Notes
- Naming warnings reported for docs examples: `top_rejection_reason_rate` and `memory_exhaustion_rate` lack the standard prefix; these are in `docs/prometheus/recording_rules.yml` and do not affect runtime rules.
