# CI Watch Validation Detail Signal Validation

## Scope

Improve `comment_evaluation_report_pr.js` so the `CI Watch Validation Report` summary stays
compact when healthy, but exposes key degraded child signals directly when the watcher validation
report is not green.

## Files

- `scripts/ci/comment_evaluation_report_pr.js`
- `tests/unit/test_comment_evaluation_report_pr_js.py`

## Change

Enhanced `summarizeCiWatchValidationReport(...)` to append detail fragments from
`ci_watch_validation_for_comment.json` when present and degraded:

- `readiness`
- `soft_smoke`
- `workflow_guardrail_summary`
- `ci_workflow_guardrail_overview`
- `evaluation_comment_support_manifest`

Example detail suffix:

```text
...; soft_smoke=exit=2, attempts=3; ci_workflow_guardrail_overview=error:status=error, ci_watch=ok, workflow_guardrail=error; evaluation_comment_support_manifest=warning:present=9/11, missing=2, invalid=0
```

## Validation

Commands run:

```bash
pytest -q tests/unit/test_comment_evaluation_report_pr_js.py
node --check scripts/ci/comment_evaluation_report_pr.js
TMPDIR=$PWD/.tmp_pytest PYTEST_ADDOPTS="--basetemp=$PWD/.tmp_pytest/basetemp" make validate-eval-with-history-ci-workflows
```

## Outcome

When `CI Watch Validation` is red, the PR comment now reveals the first failing sub-layer
directly, reducing another round-trip into the comment-support artifact JSON.
