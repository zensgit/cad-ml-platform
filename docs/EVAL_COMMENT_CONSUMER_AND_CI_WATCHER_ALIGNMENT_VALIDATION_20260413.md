# Eval Comment Consumer And CI Watcher Alignment Validation

Date: 2026-04-13

## Goal

Stabilize the evaluation PR comment consumer chain on this machine so that:

- `comment_evaluation_report_pr.js` can render the CI/report artifact summaries expected by tests.
- evaluation comment support manifest fallback logic matches the Python watcher/report consumer.
- watcher-related `Makefile` targets match the current test contract again.

## Changed Files

- `scripts/ci/comment_evaluation_report_pr.js`
- `scripts/ci/evaluation_comment_support_manifest_utils.py`
- `scripts/ci/workflow_inventory_summary_utils.py`
- `scripts/ci/generate_evaluation_comment_support_manifest.py`
- `scripts/ci/generate_ci_watcher_validation_report.py`
- `Makefile`

## What Changed

### 1. PR Comment Consumer Recovery

`scripts/ci/comment_evaluation_report_pr.js` on this machine was behind the tested contract.
It was missing:

- exported `buildEvaluationReportCommentBody`
- exported manifest / workflow inventory summary helpers
- artifact-summary fallback rendering for:
  - workflow file health
  - workflow inventory
  - workflow publish helper
  - workflow guardrail summary
  - CI workflow guardrail overview
  - CI watch validation report
  - evaluation comment support manifest

The file now:

- exports:
  - `buildEvaluationReportCommentBody`
  - `commentEvaluationReportPR`
  - `summarizeEvaluationCommentSupportManifest`
  - `summarizeWorkflowInventory`
- uses a builder-driven final comment body
- supports old env names used by tests:
  - `EVAL_COMBINED_SCORE`
  - `EVAL_VISION_SCORE`
  - `EVAL_OCR_SCORE`
  - `EVAL_MIN_COMBINED`
  - `EVAL_MIN_VISION`
  - `EVAL_MIN_OCR`
  - `SECURITY_STATUS`
- falls back to `context.runId` when `GITHUB_RUN_ID` is absent

### 2. Shared Fallback Summary Alignment

Python helpers now centralize fallback summary behavior:

- `scripts/ci/evaluation_comment_support_manifest_utils.py`
- `scripts/ci/workflow_inventory_summary_utils.py`

This keeps:

- `generate_evaluation_comment_support_manifest.py`
- `generate_ci_watcher_validation_report.py`
- `comment_evaluation_report_pr.js`

aligned on the same summary semantics.

### 3. Watcher Makefile Drift Repair

`Makefile` was missing watcher/guardrail arguments and targets required by tests.

Added or restored:

- `CI_WATCH_REPO`
- `CI_WATCH_PRINT_FAILURE_DETAILS`
- `CI_WATCH_FAILURE_DETAILS_MAX_RUNS`
- CI watcher validation report args for:
  - `--soft-smoke-summary-json`
  - `--soft-smoke-summary-md`
  - `--workflow-guardrail-summary-json`
  - `--ci-workflow-guardrail-overview-json`
  - `--evaluation-comment-support-manifest-json`
  - `--output-json`
- target:
  - `generate-ci-workflow-guardrail-overview`
- validation target:
  - `validate-generate-ci-workflow-guardrail-overview`
- `validate-ci-watchers` now includes the new guardrail overview validation step

## Validation

### Focused JS Consumer Validation

```bash
.venv/bin/python -m pytest -q tests/unit/test_comment_evaluation_report_pr_js.py
```

Result:

- `26 passed in 1.32s`

### Watcher Makefile And Guardrail Overview Validation

```bash
.venv/bin/python -m pytest -q \
  tests/unit/test_watch_commit_workflows_make_target.py \
  tests/unit/test_generate_ci_workflow_guardrail_overview.py
```

Result:

- `18 passed, 2 warnings in 0.40s`

### Broader Related Regression

```bash
.venv/bin/python -m pytest -q \
  tests/unit/test_comment_evaluation_report_pr_js.py \
  tests/unit/test_generate_ci_watcher_validation_report.py \
  tests/unit/test_generate_evaluation_comment_support_manifest.py \
  tests/unit/test_watch_commit_workflows_make_target.py \
  tests/unit/test_generate_ci_workflow_guardrail_overview.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Result:

- `66 passed, 2 warnings in 2.80s`

### Syntax Checks

```bash
node -c scripts/ci/comment_evaluation_report_pr.js
python3 -m py_compile \
  scripts/ci/evaluation_comment_support_manifest_utils.py \
  scripts/ci/workflow_inventory_summary_utils.py \
  scripts/ci/generate_evaluation_comment_support_manifest.py \
  scripts/ci/generate_ci_watcher_validation_report.py \
  scripts/ci/generate_ci_workflow_guardrail_overview.py
```

Result:

- passed

## Environment Blocker

Repository-wide `tests/unit` was also attempted, but this machine currently blocks during collection because:

- `tests/unit/test_batch_analyze_dxf_local_knowledge_context.py`
- imports `scripts/batch_analyze_dxf_local.py`
- which exits when `fastapi.testclient` is unavailable

Observed failure:

- `SystemExit: FastAPI TestClient import failed: No module named 'fastapi'`

This blocker is environment-related and outside the files changed in this repair.

## Notes

- The current warning comes from `scripts/ci/generate_ci_workflow_guardrail_overview.py` using `datetime.utcnow()`.
- That warning does not fail the current suite, but it should be cleaned up in a separate low-risk pass.
