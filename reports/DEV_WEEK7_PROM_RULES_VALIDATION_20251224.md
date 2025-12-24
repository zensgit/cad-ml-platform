# DEV_WEEK7_PROM_RULES_VALIDATION_20251224

## Scope
- Validate Prometheus rules syntax using Docker promtool.
- Fix rule validation helper to handle Docker entrypoint and non-string expressions.
- Accept standard recording rule naming convention (prefix:name).

## Changes
- `scripts/validate_prom_rules.py`
  - Docker promtool uses `--entrypoint promtool`.
  - Handles numeric/None expressions safely.
  - Allows `cad_ml:` recording rule naming convention.

## Validation
- Command: `python3 scripts/validate_prom_rules.py --rules-file prometheus/rules/cad_ml_phase5_alerts.yaml`
  - Result: Validation passed.
- Command: `python3 scripts/validate_prom_rules.py --rules-file prometheus/rules/cad_ml_recording_rules.yml`
  - Result: Validation passed.

## Notes
- promtool output did not include explicit group/rule counts; the report shows 0 while syntax validation still passed.
