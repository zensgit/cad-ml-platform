# Workflow File Health Fallback + Stress Wiring Validation (2026-03-15)

## Context
- `scripts/ci/check_workflow_file_issues.py` and `stress-tests.yml` were introduced to guard workflow-file health.
- Local `make validate-workflow-file-health` could fail in `auto` mode when `gh workflow view ... --ref HEAD` cannot resolve some workflow files on the remote ref.

## Root Cause
- `auto` mode only fell back to YAML parser for auth/token failures.
- For local-unpushed-head style errors (`could not find workflow file ... on HEAD`), it returned failure instead of graceful YAML fallback.

## Changes

### 1) Fallback strategy hardening
- File: `scripts/ci/check_workflow_file_issues.py`
- Added `_is_missing_workflow_on_ref_error(...)`.
- `--mode auto` now falls back to YAML when all `gh` failures are either:
  - auth/token related, or
  - current-ref unresolvable workflow-file errors.
- Added fallback reason: `gh_ref_unresolvable_for_local_head`.
- Warning message unified to `gh parser unavailable for current context; fallback to yaml parser`.
- `PyYAML` import changed to lazy/optional at runtime so `--mode gh` can run in minimal CI jobs without installing YAML deps.

### 2) Stress workflow wiring regression tests
- File: `tests/unit/test_stress_workflow_workflow_file_health.py`
- Coverage:
  - push path triggers include workflow-health inputs
  - `workflow-file-health` job exists and runs expected script flags
  - upload-artifact wiring (exact policy-approved SHA assertion)
  - downstream `needs: workflow-file-health` for `metrics-consistency` and `stress-unit-tests`
  - upload action pin aligned with repository policy:
    `actions/upload-artifact@bbbca2ddaa5d8feaa63e36b76fdaad77386f024f`

### 3) Make target wiring regression tests
- File: `tests/unit/test_workflow_file_health_make_target.py`
- Coverage:
  - `validate-workflow-file-health` command args
  - `validate-workflow-file-health-tests` includes expected test files
  - `validate-ci-watchers` includes `validate-workflow-file-health-tests`

### 4) Script tests expanded
- File: `tests/unit/test_check_workflow_file_issues.py`
- Added case:
  - `auto` fallback on missing-workflow-for-ref error, with summary assertion on `fallback_reason`.
  - `gh` mode works even when YAML dependency is unavailable.

## Validation Executed

### Targeted unit tests
```bash
pytest -q tests/unit/test_check_workflow_file_issues.py \
  tests/unit/test_stress_workflow_workflow_file_health.py \
  tests/unit/test_workflow_file_health_make_target.py
```
- Result: `12 passed`

### Local workflow health command
```bash
make validate-workflow-file-health
```
- Result: passed
- Observed behavior: auto mode printed fallback warning and validated all workflow files via YAML parser.

### Full watcher stack regression
```bash
make validate-ci-watchers
```
- Result: passed
- Aggregated passing suites include:
  - gh readiness checks
  - commit workflow watcher tests
  - watcher report generator tests
  - workflow-file-health tests
  - archive dispatcher tests
  - eval-with-history CI regression gates
  - Graph2D strict e2e dispatcher tests

### Action pin policy check
```bash
.venv/bin/python scripts/ci/check_workflow_action_pins.py \
  --workflows-dir .github/workflows \
  --policy-json config/workflow_action_pin_policy.json \
  --require-policy-for-all-external
```
- Result: passed (`violations_count = 0`)
- Note: `stress-tests.yml` upload step pin was aligned to policy SHA:
  `actions/upload-artifact@bbbca2ddaa5d8feaa63e36b76fdaad77386f024f`.

## Outcome
- `workflow-file-health` guard is now usable in both CI and local developer contexts.
- Local validation no longer fails spuriously due to ref-resolution-only `gh` errors in `auto` mode.
- Wiring is protected by dedicated unit tests and integrated into watcher-stack validation.
