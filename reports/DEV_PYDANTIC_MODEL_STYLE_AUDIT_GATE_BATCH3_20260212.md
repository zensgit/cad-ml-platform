# DEV_PYDANTIC_MODEL_STYLE_AUDIT_GATE_BATCH3_20260212

## Summary

Implemented Pydantic model style consistency audit (Batch 3) with CI gate,
summary visibility, baseline control, and one real style fix in API model code.

## Changes

### 1) New model-style audit gate

- Added script: `scripts/ci/audit_pydantic_model_style.py`
  - AST-based checks for:
    - `dict_model_config` (`model_config = { ... }`)
    - `mutable_literal_default` (e.g. `field = []` / `{}` in BaseModel class)
    - `mutable_field_default` (e.g. `Field(default=[])`)
    - `non_optional_none_default` (e.g. `x: int = None`)
  - Supports `--write-baseline` and `--check-regression`.

- Added baseline: `config/pydantic_model_style_baseline.json`
  - Current baseline: all counts `0`, `total_findings: 0`.

- Added tests:
  - `tests/unit/test_pydantic_model_style_audit.py`
  - `tests/unit/test_pydantic_style_audit_summary.py`

- Added summary renderer:
  - `scripts/ci/summarize_pydantic_style_audit.py`

### 2) Makefile & CI integration

- Added Make targets:
  - `audit-pydantic-style`
  - `audit-pydantic-style-regression`

- Updated `.github/workflows/ci.yml` (`lint-type` job):
  - Run style regression gate with log tee to `/tmp/pydantic-style-audit.log`
  - Append markdown summary to `GITHUB_STEP_SUMMARY`
  - Upload style audit log artifact

### 3) Real style fix found during rollout

- Updated `src/api/v1/health.py`:
  - `V16SpeedModeResponse.available_modes` changed from mutable list literal default
    to `Field(default_factory=...)`.

## Validation

- `pytest tests/unit/test_pydantic_model_style_audit.py tests/unit/test_pydantic_style_audit_summary.py tests/unit/test_pydantic_v2_audit.py tests/unit/test_pydantic_v2_audit_summary.py -q`
  - Result: `10 passed`

- `make audit-pydantic-style-regression`
  - Result: passed (`total_findings: 0`)

- `make audit-pydantic-v2-regression`
  - Result: passed (`dict_model_config: 0`, `total_findings: 0`)

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

Pydantic consistency now has two complementary gates:

- API compatibility-pattern audit (`audit_pydantic_v2.py`)
- Model style audit (`audit_pydantic_model_style.py`)

Both are baseline-controlled and CI-visible, and one concrete mutable-default risk
was removed during rollout.
