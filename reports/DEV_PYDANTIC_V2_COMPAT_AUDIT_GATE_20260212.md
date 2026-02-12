# DEV_PYDANTIC_V2_COMPAT_AUDIT_GATE_20260212

## Summary

Added a Pydantic v2 compatibility audit gate to prevent re-introducing known
v1-style patterns while keeping current codebase status explicit and traceable.

## Changes

- Added script: `scripts/ci/audit_pydantic_v2.py`
  - Scans `src` python files for high-signal v1 compatibility patterns:
    - `from pydantic.v1 import ...`
    - `@validator(...)`
    - `@root_validator(...)`
    - `class Config:`
    - `.parse_obj(...)`
    - `.parse_raw(...)`
    - `.from_orm(...)`
    - `.__fields__`
  - Supports baseline generation (`--write-baseline`) and regression check
    (`--check-regression`).

- Added baseline: `config/pydantic_v2_audit_baseline.json`
  - Current baseline on `src`: `total_findings = 0`.

- Added unit tests: `tests/unit/test_pydantic_v2_audit.py`
  - Coverage for finding collection, regression diff, and baseline payload.

- Added Make targets in `Makefile`:
  - `audit-pydantic-v2`
  - `audit-pydantic-v2-regression`

- CI integration in `.github/workflows/ci.yml`:
  - `lint-type` job now runs `make audit-pydantic-v2-regression`.

## Validation

- `.venv/bin/python -m pytest tests/unit/test_pydantic_v2_audit.py -q`
  - Result: `3 passed`

- `make audit-pydantic-v2-regression`
  - Result: passed
  - Output summary: all tracked patterns `0`, `total_findings: 0`

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

Pydantic v2 compatibility debt now has an explicit baseline and CI regression
guard, reducing risk of silent v1-pattern reintroduction.
