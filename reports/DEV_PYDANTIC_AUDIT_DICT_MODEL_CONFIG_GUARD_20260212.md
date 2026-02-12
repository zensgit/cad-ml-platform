# DEV_PYDANTIC_AUDIT_DICT_MODEL_CONFIG_GUARD_20260212

## Summary

Extended the Pydantic v2 compatibility audit to explicitly guard against
reintroducing dict-style `model_config = { ... }` usage.

## Changes

- Updated `scripts/ci/audit_pydantic_v2.py`
  - Added new audit pattern: `dict_model_config` (`model_config = {`).

- Updated baseline: `config/pydantic_v2_audit_baseline.json`
  - Added `dict_model_config` with baseline count `0`.

- Updated tests: `tests/unit/test_pydantic_v2_audit.py`
  - Added assertions for the new pattern in summary and baseline payload checks.

## Validation

- `pytest tests/unit/test_pydantic_v2_audit.py tests/unit/test_pydantic_v2_audit_summary.py -q`
  - Result: `5 passed`

- `make audit-pydantic-v2-regression`
  - Result: passed
  - Evidence:
    - `dict_model_config: 0`
    - `total_findings: 0`

- `make validate-openapi`
  - Result: `4 passed`

- `make validate-core-fast`
  - Result: passed
  - Evidence:
    - tolerance suite: `48 passed`
    - openapi/route suite: `4 passed`
    - service-mesh suite: `103 passed`
    - provider-core suite: `59 passed`
    - provider-contract suite: `4 passed, 20 deselected`

## Outcome

The audit gate now blocks both legacy v1 API patterns and non-idiomatic dict
`model_config` regressions, improving long-term Pydantic v2 consistency.
