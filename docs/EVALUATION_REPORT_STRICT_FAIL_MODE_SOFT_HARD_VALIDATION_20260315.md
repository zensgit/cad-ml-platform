# Evaluation Report Strict Fail Mode (hard/soft) Validation (2026-03-15)

## Background

`evaluation-report.yml` already had multiple strict gate fail steps (Graph2D review, Hybrid blind, Hybrid calibration, Hybrid superpass).  
For non-blocking diagnostics and staged rollout, we added a unified strict fail mode switch:

- `hard` (default): gate failure blocks workflow (existing behavior)
- `soft`: gate failure downgrades to warning and writes summary

## Implemented Changes

### 1) Workflow dispatch + env control

Updated `.github/workflows/evaluation-report.yml`:

- Added `workflow_dispatch.inputs.strict_fail_mode`
- Added global env `EVALUATION_STRICT_FAIL_MODE` (default: `hard`)

### 2) Strict mode resolver step

Added step before strict fail exits:

- `name: Resolve strict gate fail mode`
- `id: strict_fail_mode`
- Accepts aliases (`soft`, `warn`, `warning`, `nonblocking`, `non-blocking`)
- Normalizes output to `mode=hard|soft`

### 3) Soft-mode warning step

Added:

- `name: Emit strict gate warning in soft mode`
- Emits `::warning` and appends soft-mode gate status summary into `GITHUB_STEP_SUMMARY`

### 4) Gate fail guards updated

Updated strict fail `if` guards to honor soft mode:

- Hybrid superpass strict fail
- Hybrid superpass structure validation strict fail
- Graph2D review gate strict fail
- Hybrid blind gate strict fail
- Hybrid calibration gate strict fail

All now require:

```yaml
&& steps.strict_fail_mode.outputs.mode != 'soft'
```

### 5) Naming/structure fix

Fixed a regression introduced during refactor:

- first strict-mode step was incorrectly named as a fail step
- renamed to `Resolve strict gate fail mode` to avoid duplicate semantic names and test ambiguity

## Tests Updated

Updated unit tests to reflect new strict-mode behavior:

- `tests/unit/test_evaluation_report_workflow_hybrid_superpass_step.py`
- `tests/unit/test_hybrid_superpass_workflow_integration.py`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

Coverage additions:

- asserts `strict_fail_mode` dispatch input exists
- asserts `EVALUATION_STRICT_FAIL_MODE` env exists
- asserts resolver step presence and script key logic
- asserts strict fail `if` expressions include soft-mode guard

## Validation Commands and Results

### Targeted strict-mode regression tests

```bash
pytest -q \
  tests/unit/test_evaluation_report_workflow_hybrid_superpass_step.py \
  tests/unit/test_hybrid_superpass_workflow_integration.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Result: `13 passed`

### CI workflow regression gate suite

```bash
make validate-eval-with-history-ci-workflows
```

Result: `19 passed`

## Rollout Recommendation

1. Keep default `hard` in repo variables.
2. Use `workflow_dispatch` with `strict_fail_mode=soft` for staged verification runs.
3. After stable period, keep hard mode for branch protection runs, and use soft mode for exploratory/manual dispatch only.
