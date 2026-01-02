# Metrics Cardinality Audit (Markdown) (2026-01-01)

## Scope

- Generate Markdown cardinality audit report from Prometheus.

## Command

- `./.venv/bin/python scripts/cardinality_audit.py --prometheus-url http://localhost:9091 --format markdown --output reports/cardinality_audit_20260101.md`

## Results

- OK: report generated at `reports/cardinality_audit_20260101.md` (0 high-cardinality metrics).
