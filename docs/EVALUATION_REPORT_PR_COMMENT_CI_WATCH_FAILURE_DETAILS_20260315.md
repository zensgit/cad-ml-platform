# Evaluation Report PR Comment + CI Watch Failure Details (2026-03-15)

## Goal

Inject CI watcher failure details into Evaluation Report PR comments, without breaking existing workflow behavior when watcher summary is not provided.

## Implemented

1. Workflow env wiring
- File: `.github/workflows/evaluation-report.yml`
- Added env:
  - `CI_WATCH_SUMMARY_JSON_FOR_COMMENT` (defaults to empty)

2. PR comment script enhancement
- File: `.github/workflows/evaluation-report.yml` (`Comment PR with results` step)
- Added optional parsing logic (Node runtime in `actions/github-script`):
  - Reads `CI_WATCH_SUMMARY_JSON_FOR_COMMENT` if provided.
  - Parses `failure_details` and `counts.failed`.
  - Builds compact triage summary (`failed`, `reason`, top run/job/step).
  - Graceful fallback on missing file / parse error.
- Injected into PR comment tables:
  - Additional Analysis row: `CI Watch Failure Details`
  - Signal Lights row: `CI Watcher`

## Regression Coverage

- Updated `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py` assertions:
  - env contains `CI_WATCH_SUMMARY_JSON_FOR_COMMENT`
  - PR comment script includes CI watcher summary parsing and rows

## Validation Commands

```bash
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
pytest -q tests/unit/test_generate_ci_watcher_validation_report.py tests/unit/test_watch_commit_workflows.py
make validate-eval-with-history-ci-workflows
make validate-generate-ci-watch-validation-report
```

## Validation Result

- workflow graph2d extension tests: passed
- watcher/report related unit tests: passed
- eval_with_history CI workflow regression suite: passed
- watcher validation-report suite: passed

## Outcome

- PR comments now support optional direct surfacing of CI watcher failure diagnostics.
- If watcher summary is unavailable, behavior remains non-blocking and backward-compatible.
