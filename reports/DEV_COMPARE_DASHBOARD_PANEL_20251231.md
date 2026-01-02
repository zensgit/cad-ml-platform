# Compare Requests Grafana Panel (2025-12-31)

## Scope

- Add Grafana panel for `/api/compare` request rate by status.

## Changes

- `grafana/dashboards/observability.json`: new panel `Compare Requests (rate)` using `compare_requests_total`.

## Tests

```bash
.venv/bin/python -m pytest tests/test_observability_suite.py -k GrafanaDashboard -v
```

Result:
- `2 passed, 19 deselected in 10.88s`
