# DEV_DEDUP2D_METRICS_CONTRACT_UPDATE_20260101

## Design Summary
- Extended metrics contract to include dedup2d storage metrics label schemas.
- Added contract test that triggers a local storage operation and asserts storage metrics are exposed via `/metrics`.

## Files Updated
- `tests/test_metrics_contract.py`

## Verification
Planned:
```bash
pytest tests/test_metrics_contract.py -k dedup2d_storage_metrics_exposed -v
```
Notes:
- Local runtime lacks `prometheus_client`, so `/metrics` may return the fallback payload; run in the full test environment to validate.
