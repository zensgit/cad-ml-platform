# Hybrid Superpass GH Automation Development & Validation (2026-03-13)

## Scope
- Add executable GitHub automation for Hybrid superpass workflow dispatch.
- Add executable GitHub variable apply tool for superpass defaults.
- Wire new Make targets and regression coverage.

## Implemented

### 1) New GH dispatch tool
- Added `scripts/ci/dispatch_hybrid_superpass_workflow.py`.
- Required dispatched inputs:
  - `hybrid_superpass_enable=true`
  - `hybrid_superpass_missing_mode` (default `fail`)
  - `hybrid_superpass_fail_on_failed` (default `true`)
- Optional pass-through inputs (only when provided):
  - `hybrid_blind_enable`
  - `hybrid_blind_dxf_dir`
  - `hybrid_blind_fail_on_gate_failed`
  - `hybrid_blind_strict_require_real_data`
  - `hybrid_calibration_enable`
  - `hybrid_calibration_input_csv`
- Supports:
  - `--workflow`, `--ref`, `--repo`
  - `--expected-conclusion`
  - `--wait-timeout-seconds`, `--poll-interval-seconds`, `--list-limit`
  - `--output-json`, `--print-only`
- Added remote compatibility pre-check:
  - when `--repo` is provided, script checks remote workflow contains required
    inputs (`hybrid_superpass_enable`, `hybrid_superpass_missing_mode`,
    `hybrid_superpass_fail_on_failed`) before dispatch
  - `--skip-remote-input-check` can bypass this guard
- Output JSON includes:
  - `dispatch_command`
  - `run_id`
  - `conclusion`
  - `matched_expectation`
  - `overall_exit_code`

### 2) New GH vars apply tool
- Added `scripts/ci/apply_hybrid_superpass_gh_vars.py`.
- Preview mode (default) + apply mode (`--apply`).
- Recommended vars:
  - `HYBRID_SUPERPASS_ENABLE=true`
  - `HYBRID_SUPERPASS_MISSING_MODE=fail`
  - `HYBRID_SUPERPASS_FAIL_ON_FAILED=true`
  - `HYBRID_SUPERPASS_CONFIG=config/hybrid_superpass_targets.yaml` (customizable via `--config-path`)

### 3) Makefile integration
- Added targets:
  - `hybrid-superpass-e2e-gh`
  - `hybrid-superpass-apply-gh-vars`
  - `validate-hybrid-superpass-workflow`
- Added variable group for e2e/apply configuration under `HYBRID_SUPERPASS_*`.
- Added new tests into `validate-hybrid-blind-workflow` umbrella target.

### 4) Workflow behavior refinement
- `evaluation-report.yml` keeps superpass gate step as `if: always()` to support independent superpass execution path.
- `workflow_dispatch` input count has been reduced to GitHub limit (`25`) to avoid
  dispatch-time HTTP 422 (`you may only define up to 25 inputs`).

## Tests
Added:
- `tests/unit/test_dispatch_hybrid_superpass_workflow.py`
- `tests/unit/test_apply_hybrid_superpass_gh_vars.py`

Updated:
- `tests/unit/test_hybrid_calibration_make_targets.py`

## Validation Run

Executed:

```bash
pytest -q \
  tests/unit/test_dispatch_hybrid_superpass_workflow.py \
  tests/unit/test_apply_hybrid_superpass_gh_vars.py \
  tests/unit/test_hybrid_calibration_make_targets.py \
  tests/unit/test_check_hybrid_superpass_targets.py \
  tests/unit/test_hybrid_superpass_workflow_integration.py \
  tests/unit/test_evaluation_report_workflow_hybrid_superpass_step.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Result:
- `51 passed`

Command-level smoke:

```bash
make hybrid-superpass-e2e-gh HYBRID_SUPERPASS_E2E_PRINT_ONLY=1 HYBRID_SUPERPASS_E2E_REPO=zensgit/cad-ml-platform
```

- Printed executable dispatch/list/watch/view commands.

```bash
make hybrid-superpass-apply-gh-vars HYBRID_SUPERPASS_APPLY_REPO=zensgit/cad-ml-platform
```

- Printed recommended variable plan (no mutation, `apply=false`).

Remote dispatch attempt (with apply-enabled vars):

```bash
make hybrid-superpass-e2e-gh \
  HYBRID_SUPERPASS_E2E_REPO=zensgit/cad-ml-platform \
  HYBRID_SUPERPASS_E2E_EXPECTED_CONCLUSION=failure \
  HYBRID_SUPERPASS_E2E_PRINT_ONLY=0
```

- Result: HTTP 422 `Unexpected inputs provided: ["hybrid_superpass_enable", "hybrid_superpass_fail_on_failed", "hybrid_superpass_missing_mode"]`.
- Meaning: remote default-branch workflow has not yet recognized the new superpass dispatch inputs.
- Action needed: sync/merge updated `.github/workflows/evaluation-report.yml` before running superpass dispatch on remote.

With remote pre-check enabled, this mismatch is now detected before dispatch and returns
an actionable error instead of relying on HTTP 422 post-dispatch.

Branch-ref E2E after sync (`feat/hybrid-blind-drift-autotune-e2e`):

```bash
make hybrid-superpass-e2e-gh \
  HYBRID_SUPERPASS_E2E_REPO=zensgit/cad-ml-platform \
  HYBRID_SUPERPASS_E2E_REF=feat/hybrid-blind-drift-autotune-e2e \
  HYBRID_SUPERPASS_E2E_EXPECTED_CONCLUSION=failure \
  HYBRID_SUPERPASS_E2E_PRINT_ONLY=0
```

- Run URL: `https://github.com/zensgit/cad-ml-platform/actions/runs/23037797407`
- Conclusion: `failure`
- Expected: `failure`
- `matched_expectation=true`
- Failure reason in logs: `superpass_failed_under_strict_mode`

## Next Action
- For strict enforcement in GitHub Actions:
  1. `make hybrid-superpass-apply-gh-vars HYBRID_SUPERPASS_APPLY_REPO=<owner/repo> HYBRID_SUPERPASS_APPLY_EXECUTE=1`
  2. `make hybrid-superpass-e2e-gh HYBRID_SUPERPASS_E2E_REPO=<owner/repo> HYBRID_SUPERPASS_E2E_PRINT_ONLY=0`
