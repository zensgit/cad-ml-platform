# Workflow Identity Invariants Alignment Development Plan

Date: 2026-04-14
Owner: Codex
Scope: Align `scripts/ci/check_workflow_identity_invariants.py` with the current workflow contracts for `evaluation-report.yml` and `hybrid-superpass-e2e.yml`.

## Background

`check_workflow_identity_invariants.py` had drifted behind the actual GitHub Actions workflow definitions:

- `evaluation-report.yml` intentionally exposes `workflow_dispatch: {}` and uses `env` / repository variables for compatibility, but the checker still required legacy dispatch inputs.
- `hybrid-superpass-e2e.yml` had been simplified to eight current dispatch inputs, while the checker still expected older wrapper-only inputs such as `ref` and `expected_conclusion`.

This caused the identity checker to fail even though the workflows themselves were already correct.

## Goals

1. Restore `check_workflow_identity_invariants.py` to green against the real workflow files.
2. Keep the fix minimal by updating the checker spec, not the workflow YAMLs.
3. Preserve watcher-required validation for canonical workflows, including `Evaluation Report` and `Governance Gates`.
4. Add or update tests so the expected CI watch mapping reflects the current required workflow set.

## Implementation Plan

### 1. Update identity specs

Change `scripts/ci/check_workflow_identity_invariants.py`:

- `evaluation-report.yml`
  - Keep expected workflow name as `Evaluation Report`
  - Remove legacy `required_inputs`
  - Keep `require_ci_watch=True`

- `hybrid-superpass-e2e.yml`
  - Replace legacy required inputs with the current dispatch contract:
    - `hybrid_superpass_enable`
    - `hybrid_superpass_missing_mode`
    - `hybrid_superpass_fail_on_failed`
    - `hybrid_blind_gate_report_json`
    - `hybrid_calibration_json`
    - `hybrid_superpass_config`
    - `hybrid_superpass_output_json`
    - `dispatch_trace_id`

### 2. Update unit-test fixture data

Change `tests/unit/test_check_workflow_identity_invariants.py`:

- Update the sample `CI_WATCH_REQUIRED_WORKFLOWS` CSV to include `Governance Gates`
- Leave the existing negative tests in place for:
  - name mismatch
  - missing required input
  - unexpected `.yaml` twin
  - missing required CI-watch workflow

### 3. Verification

Run:

```bash
.venv311/bin/python scripts/ci/check_workflow_identity_invariants.py
.venv311/bin/python -m pytest -q tests/unit/test_check_workflow_identity_invariants.py
.venv311/bin/python scripts/ci/generate_workflow_inventory_report.py --output-json /tmp/workflow_inventory_report.json --output-md /tmp/workflow_inventory_report.md
```

### 4. Sidecar review

Use `Claude Code CLI` as read-only sidecar review only. Do not make the validation path depend on it.

## Expected Outcome

- `workflow identity` checks pass locally
- Inventory generation still reports `Evaluation Report` and `Governance Gates` as required and healthy
- No workflow YAML changes are needed
