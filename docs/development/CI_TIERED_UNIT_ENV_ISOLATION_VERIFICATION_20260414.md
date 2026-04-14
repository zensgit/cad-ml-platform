# CI Tiered Unit Env Isolation Verification

Date: 2026-04-14
PR: `#398`
Head: `submit/local-main-20260414`

## Remote Failure Reference

Failing workflow before fix:

- `CI Tiered Tests`
- run: `24399473960`
- failing job: `unit-tier`
- failing test:
  - `tests/unit/test_comment_evaluation_report_pr_js.py::test_comment_evaluation_report_pr_js_runtime_body_matches_builder_output`

## Local Verification

### 1. Targeted failing test with intentionally polluted environment

Command:

```bash
env \
  GITHUB_EVENT_INPUTS_JSON='{"min_combined":"0.99"}' \
  GITHUB_STEPS_JSON='{"graph2d_review_gate":{"outputs":{"status":"passed","exit_code":"0","headline":"ok"}}}' \
  CI_WATCH_SUMMARY_JSON_FOR_COMMENT='/tmp/leak-ci-watch.json' \
  WORKFLOW_FILE_HEALTH_SUMMARY_JSON_FOR_COMMENT='/tmp/leak-workflow-health.json' \
  WORKFLOW_INVENTORY_REPORT_JSON_FOR_COMMENT='/tmp/leak-workflow-inventory.json' \
  EVALUATION_COMMENT_SUPPORT_MANIFEST_JSON_FOR_COMMENT='/tmp/leak-comment-support.json' \
  EVAL_REPORTING_STACK_SUMMARY_JSON_FOR_COMMENT='/tmp/leak-stack.json' \
  .venv311/bin/python -m pytest -q \
    tests/unit/test_comment_evaluation_report_pr_js.py::test_comment_evaluation_report_pr_js_runtime_body_matches_builder_output \
    -vv
```

Result:

- `1 passed`

### 2. Full evaluation-report comment JS unit file

Command:

```bash
.venv311/bin/python -m pytest -q tests/unit/test_comment_evaluation_report_pr_js.py
```

Result:

- `26 passed`

### 3. Tiered unit entrypoint under polluted environment

Command:

```bash
env \
  GITHUB_EVENT_INPUTS_JSON='{"min_combined":"0.99"}' \
  GITHUB_STEPS_JSON='{"graph2d_review_gate":{"outputs":{"status":"passed","exit_code":"0","headline":"ok"}}}' \
  bash scripts/test_with_local_api.sh --suite unit
```

Status:

- running during verification to confirm the unit-tier path no longer trips on the JS comment drift

## Additional Parallel Findings

Read-only parallel review confirmed:

- `Action Pin Guard` failure is not caused by this batch
- root cause there is repository-wide workflow action refs still pinned to tags instead of SHAs
- this remains a separate remediation stream

## Conclusion

This fix addresses the real blocker in `CI Tiered Tests`:

- deterministic isolation of Node subprocess environment for `comment_evaluation_report_pr` unit coverage

The production script was not changed; only the test harness was hardened.
