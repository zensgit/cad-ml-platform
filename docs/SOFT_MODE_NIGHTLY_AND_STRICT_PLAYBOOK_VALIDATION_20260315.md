# Soft-Mode Nightly + Strict Playbook Validation (2026-03-15)

## Scope

Parallel delivery for two acceleration items:

1. Add `validate-soft-mode-smoke` automation and nightly CI workflow integration.
2. Add strict-gate reason-to-playbook mapping in Evaluation Report PR comments.

## Implemented

### A) Soft-mode smoke automation

- New Make target:
  - `validate-soft-mode-smoke`
- New Make validation target:
  - `validate-soft-mode-smoke-workflow`
- New nightly workflow:
  - `.github/workflows/evaluation-soft-mode-smoke.yml`
  - triggers:
    - `schedule`: `20 3 * * *`
    - `workflow_dispatch`
  - behavior:
    - run `scripts/ci/dispatch_evaluation_soft_mode_smoke.py`
    - upload summary artifact
    - append result to `GITHUB_STEP_SUMMARY`

### B) Strict reason -> playbook mapping

- Updated PR comment module:
  - `scripts/ci/comment_evaluation_report_pr.js`
- New playbook doc:
  - `docs/STRICT_GATE_PLAYBOOK.md`
- Added outputs in PR comment:
  - `Strict Gate Playbook`
  - `Strict Gate Decision Path` includes playbook links

### C) Tests and wiring updates

- Added:
  - `tests/unit/test_evaluation_soft_mode_smoke_workflow.py`
- Updated:
  - `tests/unit/test_hybrid_calibration_make_targets.py`
  - `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
- `validate-hybrid-superpass-workflow` now includes soft-mode smoke tests.

## Verification

### Unit and integration regressions

```bash
pytest -q \
  tests/unit/test_dispatch_evaluation_soft_mode_smoke.py \
  tests/unit/test_evaluation_soft_mode_smoke_workflow.py \
  tests/unit/test_hybrid_calibration_make_targets.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/unit/test_evaluation_report_workflow_hybrid_superpass_step.py \
  tests/unit/test_hybrid_superpass_workflow_integration.py
```

Result: `45 passed`

```bash
make validate-soft-mode-smoke-workflow
```

Result: `32 passed`

```bash
make validate-hybrid-superpass-workflow
```

Result: `68 passed`

### Syntax checks

```bash
node --check scripts/ci/comment_evaluation_report_pr.js
python3 -m py_compile scripts/ci/dispatch_evaluation_soft_mode_smoke.py
```

Result: passed.

### Remote execution notes

- Real soft-mode smoke dispatch via script has been verified in this branch:
  - run id: `23110740927` (after this change set)
  - conclusion: `success`
  - strict soft marker detected and variable restore succeeded.
- Previous baseline run before this change set: `23110519585`.
- Direct `gh workflow run evaluation-soft-mode-smoke.yml` currently returns 404 before merge because GitHub requires the workflow file to exist on the default branch for dispatch lookup.
