# Eval Comment Support And CI Watch Integration Validation

## Scope

- Surface `evaluation_comment_support_manifest.json` in the evaluation PR comment.
- Feed the same manifest into `generate_ci_watcher_validation_report.py`.
- Wire the manifest path through `evaluation-report.yml` and `Makefile`.

## Changed Files

- `scripts/ci/comment_evaluation_report_pr.js`
- `scripts/ci/generate_ci_watcher_validation_report.py`
- `.github/workflows/evaluation-report.yml`
- `Makefile`
- `tests/unit/test_comment_evaluation_report_pr_js.py`
- `tests/unit/test_generate_ci_watcher_validation_report.py`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
- `tests/unit/test_watch_commit_workflows_make_target.py`

## Validation Commands

```bash
pytest -q \
  tests/unit/test_comment_evaluation_report_pr_js.py \
  tests/unit/test_generate_ci_watcher_validation_report.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/unit/test_watch_commit_workflows_make_target.py

TMPDIR=$PWD/.tmp_pytest \
PYTEST_ADDOPTS="--basetemp=$PWD/.tmp_pytest/basetemp" \
make validate-generate-ci-watch-validation-report

TMPDIR=$PWD/.tmp_pytest \
PYTEST_ADDOPTS="--basetemp=$PWD/.tmp_pytest/basetemp" \
make validate-eval-with-history-ci-workflows
```

## Outcome

- Evaluation PR comments now expose `Evaluation Comment Support Manifest` and `Comment Support Bundle`.
- CI watch validation summaries now include `comment_support=<status>` and an `Evaluation Comment Support Manifest` section.
- `evaluation-report.yml` passes the manifest into both the watcher validation report and the PR comment step.
- `generate-ci-watch-validation-report` now accepts `--evaluation-comment-support-manifest-json`.
