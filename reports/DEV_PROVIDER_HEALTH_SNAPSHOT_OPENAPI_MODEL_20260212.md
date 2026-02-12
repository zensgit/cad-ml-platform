# DEV_PROVIDER_HEALTH_SNAPSHOT_OPENAPI_MODEL_20260212

## Summary

Promoted `/api/v1/providers/health` result `snapshot` from an untyped
`Dict[str, Any]` to an explicit Pydantic model so the OpenAPI schema reflects
the stable snapshot keys and the OpenAPI snapshot gate can detect drift.

## Changes

- Updated `src/api/v1/health.py`
  - Added response model `ProviderStatusSnapshot` (extra fields allowed).
  - Updated `ProviderHealthItem.snapshot` -> `Optional[ProviderStatusSnapshot]`.
  - Hardened snapshot builder to always include stable keys:
    - `name`, `provider_type`, `status`, `last_error`,
      `last_health_check_at`, `last_health_check_latency_ms`

- Updated `tests/contract/test_api_contract.py`
  - Provider health response contract now asserts snapshot keys when present.
  - Added OpenAPI schema contract that snapshot is typed and contains the
    expected properties.

- Updated OpenAPI snapshot baseline
  - `config/openapi_schema_snapshot.json` regenerated via:
    - `make openapi-snapshot-update`

## Validation

- `.venv/bin/python -m pytest tests/unit/test_provider_health_endpoint.py tests/contract/test_api_contract.py -k provider_health -v`
  - Result: `5 passed`

- `make openapi-snapshot-update`
  - Result: baseline regenerated
  - Evidence: `paths=161`, `operations=166`

- `make validate-openapi`
  - Result: `5 passed`

- `make validate-core-fast`
  - Result: passed

## Outcome

Provider health snapshots are now a first-class OpenAPI contract with stable
keys, while still allowing providers to attach additional non-breaking fields.

