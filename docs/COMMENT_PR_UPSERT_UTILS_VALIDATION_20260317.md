# Comment PR Upsert Utils Validation

Date: 2026-03-17
Branch: `feat/hybrid-blind-drift-autotune-e2e`

## Scope

Extract shared GitHub PR comment create/update logic from:

- `scripts/ci/comment_evaluation_report_pr.js`
- `scripts/ci/comment_soft_mode_smoke_pr.js`

into:

- `scripts/ci/comment_pr_utils.js`

## Changes

- Added `findBotCommentByMarker(...)` and `upsertBotIssueComment(...)`.
- Switched `evaluation-report` JS comment flow to the shared helper.
- Switched `soft-mode smoke` JS comment flow to the shared helper.
- Added Node runtime tests for update-path and create-path behavior.
- Wired the new helper test into existing Make validation targets.

## Validation

Commands:

```bash
node --check scripts/ci/comment_pr_utils.js
node --check scripts/ci/comment_evaluation_report_pr.js
node --check scripts/ci/comment_soft_mode_smoke_pr.js
pytest -q \
  tests/unit/test_comment_pr_utils_js.py \
  tests/unit/test_comment_evaluation_report_pr_js.py \
  tests/unit/test_comment_soft_mode_smoke_pr_js.py \
  tests/unit/test_graph2d_parallel_make_targets.py \
  tests/unit/test_hybrid_calibration_make_targets.py
make validate-soft-mode-smoke-comment
```

Expected results:

- shared helper syntax passes
- evaluation-report JS comment tests pass
- soft-mode JS comment tests pass
- Make target assertions include `test_comment_pr_utils_js.py`
- `validate-soft-mode-smoke-comment` passes with the new helper in the chain
- `validate-eval-with-history-ci-workflows` passes with the new helper wired into the evaluation-report chain

## Notes

- During validation, `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py` still contained one brittle source-literal assertion tied to the old inline comment implementation. It was updated to assert the current structure instead:
  - summary readers remain present (`summarizeCiWatchFailure`, `summarizeWorkflowFileHealth`, `summarizeWorkflowInventory`)
  - `fsApi.existsSync(summaryPath)` guard remains present inside those readers
  - `commentEvaluationReportPR(...)` still calls all three summary readers before building the PR comment body
