# DEV_PYDANTIC_AUDIT_SUMMARY_AND_V2_MIGRATION_BATCH1_20260212

## Summary

Implemented both requested items:

1. Added CI summary visibility for Pydantic v2 audit gate.
2. Completed first migration batch on high-frequency model modules using explicit
   Pydantic v2 config types.

## Changes

### 1) CI summary for pydantic audit

- Added script: `scripts/ci/summarize_pydantic_v2_audit.py`
  - Parses audit log output and renders markdown summary table for `GITHUB_STEP_SUMMARY`.
  - Includes gate status, total findings, per-pattern counts, and log tail.

- Added tests: `tests/unit/test_pydantic_v2_audit_summary.py`
  - Covers no-regression and regression-detected scenarios.

- Updated workflow: `.github/workflows/ci.yml`
  - `lint-type` audit step now writes log to `/tmp/pydantic-v2-audit.log`.
  - Added summary append step (always runs).
  - Added artifact upload for audit log.

### 2) Pydantic v2 migration batch 1 (low risk)

- Updated `src/core/vision/base.py`
  - Replaced dict-based `model_config` with explicit `ConfigDict(...)` in:
    - `VisionAnalyzeRequest`
    - `CadFeatureStats`
    - `VisionAnalyzeResponse`

- Updated `src/core/config.py`
  - Switched from dict `model_config` to `SettingsConfigDict(...)`.

- Updated `src/core/config/__init__.py`
  - Switched from dict `model_config` to `SettingsConfigDict(...)`.

## Validation

- `pytest tests/unit/test_pydantic_v2_audit.py tests/unit/test_pydantic_v2_audit_summary.py -q`
  - Result: `5 passed`

- `make audit-pydantic-v2-regression`
  - Result: passed
  - Evidence: all tracked patterns `0`, `total_findings: 0`

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

- CI now exposes Pydantic v2 audit results in step summary and retains audit logs as artifact.
- First migration batch adopts clearer, explicit v2 config declarations without behavior changes.
