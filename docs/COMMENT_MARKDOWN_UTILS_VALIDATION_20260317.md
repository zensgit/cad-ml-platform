# Comment Markdown Utils Validation

Date: 2026-03-17

## Goal

Extract shared Markdown builders for PR comment scripts so:

- `evaluation-report` and `soft-mode smoke` comments stop hand-assembling tables/sections
- comment layout stays stable while reducing template drift
- shared helper behavior is covered by runtime tests and Make validation targets

## Added

- `scripts/ci/comment_markdown_utils.js`
- `tests/unit/test_comment_markdown_utils_js.py`

Refactored comment scripts:

- `scripts/ci/comment_evaluation_report_pr.js`
- `scripts/ci/comment_soft_mode_smoke_pr.js`

Validation targets updated:

- `validate-eval-with-history-ci-workflows`
- `validate-soft-mode-smoke-comment`

Regression assertions updated:

- `tests/unit/test_graph2d_parallel_make_targets.py`
- `tests/unit/test_hybrid_calibration_make_targets.py`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation

```bash
pytest -q \
  tests/unit/test_comment_markdown_utils_js.py \
  tests/unit/test_comment_evaluation_report_pr_js.py \
  tests/unit/test_comment_soft_mode_smoke_pr_js.py \
  tests/unit/test_graph2d_parallel_make_targets.py \
  tests/unit/test_hybrid_calibration_make_targets.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

```bash
make validate-eval-with-history-ci-workflows
make validate-soft-mode-smoke-comment
```

Results:

- `pytest -q tests/unit/test_comment_markdown_utils_js.py tests/unit/test_comment_evaluation_report_pr_js.py tests/unit/test_comment_soft_mode_smoke_pr_js.py tests/unit/test_graph2d_parallel_make_targets.py tests/unit/test_hybrid_calibration_make_targets.py tests/unit/test_evaluation_report_workflow_graph2d_extensions.py` -> `67 passed`
- `make validate-eval-with-history-ci-workflows` -> `27 passed`
- `make validate-soft-mode-smoke-comment` -> `4 passed`

## Notes

- This is a comment rendering refactor only. No workflow YAML behavior changed.
- `comment_evaluation_report_pr.js` now builds sections/tables via helper calls instead of one large inline template.
- `comment_soft_mode_smoke_pr.js` uses the same helper, so future comment-format changes can be applied once.

## Python Bridge Follow-up

Added:

- `scripts/ci/comment_markdown_utils.py`
- `tests/unit/test_comment_markdown_utils_py.py`

Refactored:

- `scripts/ci/post_soft_mode_smoke_pr_comment.py`

Validation target updated:

- `validate-soft-mode-smoke-comment-pr`

Follow-up validation:

```bash
pytest -q \
  tests/unit/test_comment_markdown_utils_py.py \
  tests/unit/test_post_soft_mode_smoke_pr_comment.py \
  tests/unit/test_hybrid_calibration_make_targets.py
```

```bash
make validate-soft-mode-smoke-comment-pr
```

Follow-up results:

- `pytest -q tests/unit/test_comment_markdown_utils_py.py tests/unit/test_post_soft_mode_smoke_pr_comment.py tests/unit/test_hybrid_calibration_make_targets.py` -> `46 passed`
- `make validate-soft-mode-smoke-comment-pr` -> `46 passed`

Follow-up notes:

- The Python PR bridge now uses the same table/section/footer composition pattern as the JS comment scripts.
- `post_soft_mode_smoke_pr_comment.py` now reuses `read_json_object(..., "summary")` from `summary_render_utils.py` instead of duplicating JSON object parsing.

## Soft-Mode Body Parity Follow-up

Goal:

- lock JS soft-mode PR comments and Python soft-mode PR bridge to the same rendered body semantics
- prevent drift in field order, attempt-line format, footer layout, and boolean rendering

Added:

- `tests/unit/test_soft_mode_comment_body_consistency.py`

Updated:

- `scripts/ci/comment_soft_mode_smoke_pr.js`
- `scripts/ci/post_soft_mode_smoke_pr_comment.py`
- `scripts/ci/comment_markdown_utils.py`
- `tests/unit/test_comment_soft_mode_smoke_pr_js.py`
- `tests/unit/test_post_soft_mode_smoke_pr_comment.py`
- `tests/unit/test_comment_markdown_utils_py.py`
- `tests/unit/test_hybrid_calibration_make_targets.py`

Validation targets updated:

- `validate-soft-mode-smoke-comment`
- `validate-soft-mode-smoke-comment-pr`

Parity validation:

```bash
pytest -q \
  tests/unit/test_comment_markdown_utils_py.py \
  tests/unit/test_comment_soft_mode_smoke_pr_js.py \
  tests/unit/test_post_soft_mode_smoke_pr_comment.py \
  tests/unit/test_soft_mode_comment_body_consistency.py \
  tests/unit/test_hybrid_calibration_make_targets.py
```

```bash
make validate-soft-mode-smoke-comment
make validate-soft-mode-smoke-comment-pr
```

Parity results:

- `pytest -q tests/unit/test_comment_markdown_utils_py.py tests/unit/test_comment_soft_mode_smoke_pr_js.py tests/unit/test_post_soft_mode_smoke_pr_comment.py tests/unit/test_soft_mode_comment_body_consistency.py tests/unit/test_hybrid_calibration_make_targets.py` -> `50 passed`
- `make validate-soft-mode-smoke-comment` -> `5 passed`
- `make validate-soft-mode-smoke-comment-pr` -> `47 passed`

Parity notes:

- Both paths now emit `dispatch_exit_code` in the summary table.
- Both paths now render attempt lines as `dispatch_exit_code=..., soft_marker_ok=..., message=...`.
- Both paths now use lowercase boolean text (`true/false`) and include the same footer separator.

## Evaluation Report Body Builder Follow-up

Goal:

- extract the final evaluation-report PR comment body into a dedicated builder
- keep env parsing, strict-decision logic, JSON summary parsing, and GitHub create/update behavior unchanged
- replace brittle source-literal checks with builder/helper wiring checks plus runtime body validation

Updated:

- `scripts/ci/comment_evaluation_report_pr.js`
- `tests/unit/test_comment_evaluation_report_pr_js.py`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

Follow-up validation:

```bash
node --check scripts/ci/comment_evaluation_report_pr.js
pytest -q \
  tests/unit/test_comment_markdown_utils_js.py \
  tests/unit/test_comment_evaluation_report_pr_js.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/unit/test_graph2d_parallel_make_targets.py
```

Results:

- `node --check scripts/ci/comment_evaluation_report_pr.js` -> `ok`
- `pytest -q tests/unit/test_comment_markdown_utils_js.py tests/unit/test_comment_evaluation_report_pr_js.py tests/unit/test_evaluation_report_workflow_graph2d_extensions.py tests/unit/test_graph2d_parallel_make_targets.py` -> `26 passed`

Notes:

- `comment_evaluation_report_pr.js` now exports `buildEvaluationReportCommentBody(...)`.
- `test_comment_evaluation_report_pr_js.py` now includes a direct runtime builder test with fixed timestamp and sha.
- `make validate-eval-with-history-ci-workflows` was rerun with workspace-local temp configuration, but is currently blocked by an unrelated existing failure in `tests/unit/test_eval_with_history_script_history_sequence.py` caused by invalid history-report JSON under the current branch state.
