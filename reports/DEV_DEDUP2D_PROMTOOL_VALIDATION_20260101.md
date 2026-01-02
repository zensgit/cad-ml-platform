# DEV_DEDUP2D_PROMTOOL_VALIDATION_20260101

## Scope
- Run full Prometheus recording rule validation using promtool (via Docker).

## Command
```bash
python3 scripts/validate_prom_rules.py
```

## Result
- Syntax validation: PASSED
- Expressions: PASSED
- Naming warnings (pre-existing):
  - `top_rejection_reason_rate` lacks standard prefix
  - `memory_exhaustion_rate` lacks standard prefix

## Notes
- Validation executed via Docker `prom/prometheus` image.
