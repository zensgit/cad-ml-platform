# CI Watch Validation JSON Comment Integration

## Scope

Promote `generate_ci_watcher_validation_report.py` from markdown-only output to dual output:

- watcher validation markdown
- watcher validation summary JSON

Then wire that JSON summary into `evaluation-report.yml` and
`comment_evaluation_report_pr.js` so PR comments can surface a stable watcher-validation
signal without parsing markdown.

## Files

- `scripts/ci/generate_ci_watcher_validation_report.py`
- `.github/workflows/evaluation-report.yml`
- `scripts/ci/comment_evaluation_report_pr.js`
- `Makefile`
- `tests/unit/test_generate_ci_watcher_validation_report.py`
- `tests/unit/test_watch_commit_workflows_make_target.py`
- `tests/unit/test_comment_evaluation_report_pr_js.py`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Change

- Added `--output-json` to `generate_ci_watcher_validation_report.py`.
- Added structured watcher-validation summary payload with:
  - `verdict`
  - `verdict_success`
  - top-level `summary`
  - section presence/status for readiness, soft-smoke, workflow guardrail, and CI workflow overview
- Extended `generate-ci-watch-validation-report` make target to pass `CI_WATCH_REPORT_OUTPUT_JSON`.
- In `evaluation-report.yml`:
  - build optional `ci_watch_validation_for_comment.json`
  - append `ci_watch_validation_for_comment.md` to `GITHUB_STEP_SUMMARY`
  - pass `CI_WATCH_VALIDATION_REPORT_JSON_FOR_COMMENT` into PR comment rendering
- In `comment_evaluation_report_pr.js`:
  - added `summarizeCiWatchValidationReport(...)`
  - added `CI Watch Validation Report` row
  - added `CI Watch Validation` signal light

## Validation

Commands run:

```bash
pytest -q tests/unit/test_generate_ci_watcher_validation_report.py tests/unit/test_watch_commit_workflows_make_target.py
pytest -q tests/unit/test_comment_evaluation_report_pr_js.py tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
node --check scripts/ci/comment_evaluation_report_pr.js
make validate-generate-ci-watch-validation-report
TMPDIR=$PWD/.tmp_pytest PYTEST_ADDOPTS="--basetemp=$PWD/.tmp_pytest/basetemp" make validate-eval-with-history-ci-workflows
```

## Outcome

`evaluation-report` now has a stable, structured watcher-validation signal available in both:

- step summary markdown
- PR comment summary rows / signal lights

This removes the need to parse markdown or reconstruct watcher-validation state inside the
comment layer.
