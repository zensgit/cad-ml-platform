# Action Pin Guard Remediation Verification

Date: 2026-04-14
Branch: `submit/local-main-20260414`
PR: `#398`

## Implemented Changes

### Workflow files updated

1. `.github/workflows/code-quality.yml`
2. `.github/workflows/evaluation-report.yml`
3. `.github/workflows/hybrid-superpass-e2e.yml`
4. `.github/workflows/hybrid-superpass-nightly.yml`

### Workflow tests updated

1. `tests/unit/test_hybrid_superpass_nightly_workflow.py`
2. `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Local Verification

### Action pin guard

Command:

```bash
python3 scripts/ci/check_workflow_action_pins.py \
  --workflows-dir .github/workflows \
  --policy-json config/workflow_action_pin_policy.json \
  --require-policy-for-all-external
```

Result:

```json
{
  "status": "ok",
  "violations_count": 0
}
```

### Workflow regression set

Command:

```bash
.venv311/bin/python -m pytest -q \
  tests/unit/test_action_pin_guard_workflow.py \
  tests/unit/test_additional_workflow_comment_helper_adoption.py \
  tests/unit/test_hybrid_superpass_e2e_workflow.py \
  tests/unit/test_hybrid_superpass_nightly_workflow.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py \
  tests/unit/test_evaluation_report_workflow_hybrid_superpass_step.py \
  tests/unit/test_evaluation_report_workflow_release_summary.py \
  tests/unit/test_evaluation_report_workflow_eval_reporting_stack.py \
  tests/unit/test_hybrid_superpass_workflow_integration.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Result:

```text
108 passed, 7 warnings
```

## Sidecar Review

`Claude Code CLI` was used as a read-only sidecar reviewer for the pin-remediation diff. Main execution and validation did not depend on it.

## Remote Verification

Pending at document creation time.

Target gate:
- `Action Pin Guard`

