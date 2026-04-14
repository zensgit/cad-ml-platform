# CI Tiered Unit Env Isolation Development Plan

Date: 2026-04-14
PR: `#398`
Head: `submit/local-main-20260414`

## Background

`CI Tiered Tests` failed in remote run `24399473960` at job `unit-tier`.

The single failing test was:

- `tests/unit/test_comment_evaluation_report_pr_js.py::test_comment_evaluation_report_pr_js_runtime_body_matches_builder_output`

The failure message was:

- `runtime comment body drifted from builder output`

Local single-test reproduction passed, which indicated environment-dependent drift rather than a deterministic bug in `scripts/ci/comment_evaluation_report_pr.js`.

## Root Cause

`tests/unit/test_comment_evaluation_report_pr_js.py` launched Node subprocesses with the full parent process environment.

That made the runtime builder/body comparison sensitive to leaked or pre-existing environment variables such as:

- `GITHUB_EVENT_INPUTS_JSON`
- `GITHUB_STEPS_JSON`
- `CI_WATCH_SUMMARY_JSON_FOR_COMMENT`
- `WORKFLOW_FILE_HEALTH_SUMMARY_JSON_FOR_COMMENT`
- `WORKFLOW_INVENTORY_REPORT_JSON_FOR_COMMENT`
- `EVALUATION_COMMENT_SUPPORT_MANIFEST_JSON_FOR_COMMENT`
- `EVAL_REPORTING_STACK_SUMMARY_JSON_FOR_COMMENT`

In GitHub-hosted runners, or after other test interactions, those variables can change the runtime-generated comment body without changing the explicit expected view model inside the test.

## Change

File changed:

- `tests/unit/test_comment_evaluation_report_pr_js.py`

Implementation:

- Added `_node_subprocess_env()` helper.
- Restricted Node subprocess inheritance to a minimal runtime-safe environment:
  - `PATH`
  - `HOME`
  - `TMPDIR`
  - `TMP`
  - `TEMP`
  - `SystemRoot`
  - `COMSPEC`
- Updated `_run_node_inline()` to use that sanitized environment.

## Why This Fix

This keeps the test deterministic while leaving the production script unchanged.

It is the smallest safe fix because:

- the failing behavior was test-environment drift
- the runtime script already matched builder output in a clean environment
- changing the business script would have risked introducing comment-format regressions for an issue isolated to test setup

## Expected Outcome

- `test_comment_evaluation_report_pr_js_runtime_body_matches_builder_output` becomes stable in GitHub-hosted runners
- `CI Tiered Tests` no longer fails on comment-body drift caused by inherited environment leakage

## Non-Goals

- No change to `scripts/ci/comment_evaluation_report_pr.js` logic
- No change to workflow comment semantics
- No attempt to fix unrelated `Action Pin Guard`, `PR Auto Label and Comment`, or security-policy failures in this batch
