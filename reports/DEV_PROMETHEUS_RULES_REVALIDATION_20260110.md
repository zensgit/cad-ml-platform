# DEV_PROMETHEUS_RULES_REVALIDATION_20260110

## Scope
Re-run Prometheus rules validation for recording and alerting rules via promtool.

## Commands
- `python3 scripts/validate_prom_rules.py --rules-file config/prometheus/recording_rules.yml`
- `python3 scripts/validate_prom_rules.py --rules-file config/prometheus/alerting_rules.yml`

## Results
- Recording rules: promtool syntax validation passed; naming warnings remain for existing `cad:`-prefixed rules.
- Alerting rules: promtool syntax validation passed.
