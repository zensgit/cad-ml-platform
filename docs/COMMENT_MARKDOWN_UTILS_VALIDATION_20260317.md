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
