# Strict-Mode Soft Smoke + PR Comment Decision Path Validation (2026-03-15)

## Scope

This update delivers two parallel improvements:

1. **Soft-mode smoke runner with automatic variable restore**
2. **PR comment strict-gate decision path and actionable guidance**

## Changes

### 1) New smoke runner

Added script:

- `scripts/ci/dispatch_evaluation_soft_mode_smoke.py`

Key behavior:

- Snapshot repo variable `EVALUATION_STRICT_FAIL_MODE`
- Temporarily set it to `soft`
- Dispatch `evaluation-report.yml` via existing dispatcher logic
- Watch run and verify log marker: `Resolved strict fail mode: soft`
- Restore previous variable state in `finally`:
  - restore previous value if it existed
  - delete variable if it was originally absent
- Emit structured JSON result

### 2) PR comment strict-gate decision path

Updated:

- `.github/workflows/evaluation-report.yml`  
  Added env passthrough for strict-mode resolution and hybrid superpass strict/validation signals.

- `scripts/ci/comment_evaluation_report_pr.js`  
  Added:
  - strict-mode parsing (`hard|soft`)
  - strict failure request aggregation (review/blind/calibration/superpass/superpass-validation)
  - decision result (`no_strict_fail_requests` / `downgraded_to_warning` / `blocking_failure_expected`)
  - `Strict Gate Policy` in Additional Analysis + Signal Lights
  - `Strict Gate Decision Path` section with recommended next actions

## Tests

Added:

- `tests/unit/test_dispatch_evaluation_soft_mode_smoke.py`

Updated:

- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
  - verifies new PR comment env passthrough keys
  - verifies strict decision-path content markers in comment module

## Validation

### Unit/Workflow regression

```bash
pytest -q \
  tests/unit/test_dispatch_evaluation_soft_mode_smoke.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/unit/test_evaluation_report_workflow_hybrid_superpass_step.py \
  tests/unit/test_hybrid_superpass_workflow_integration.py
```

Result: `17 passed`

```bash
make validate-eval-with-history-ci-workflows
```

Result: `19 passed`

### Syntax checks

```bash
node --check scripts/ci/comment_evaluation_report_pr.js
python3 -m py_compile scripts/ci/dispatch_evaluation_soft_mode_smoke.py
```

Result: passed.

### Real remote smoke run

Command:

```bash
python3 scripts/ci/dispatch_evaluation_soft_mode_smoke.py \
  --repo zensgit/cad-ml-platform \
  --ref feat/hybrid-blind-drift-autotune-e2e \
  --output-json reports/experiments/20260315/soft_mode_smoke_20260315.json \
  --wait-timeout-seconds 1200
```

Observed:

- Run ID: `23110519585`
- Run URL: `https://github.com/zensgit/cad-ml-platform/actions/runs/23110519585`
- Conclusion: `success`
- Strict marker detected: `Resolved strict fail mode: soft`
- Restore status: `restore_ok=true`

Artifact:

- `reports/experiments/20260315/soft_mode_smoke_20260315.json`

