# Metrics Cardinality Check (2026-01-01)

## Scope

- Run cardinality audit against the local Prometheus instance.

## Command

- `./.venv/bin/python scripts/cardinality_audit.py --prometheus-url http://localhost:9091 --warning-threshold 100 --critical-threshold 1000 --format json --output reports/cardinality_report.json`

## Results

- OK: 508 metrics discovered; 231 analyzed; 0 high-cardinality metrics.
- Report saved to `reports/cardinality_report.json`.
