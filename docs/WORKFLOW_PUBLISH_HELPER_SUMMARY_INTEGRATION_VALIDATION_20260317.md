# Workflow Publish Helper Summary Integration Validation

Date: 2026-03-17
Branch: `feat/hybrid-blind-drift-autotune-e2e`

## Scope

Integrate workflow publish-helper adoption summary into:

- `.github/workflows/stress-tests.yml`
- `.github/workflows/evaluation-report.yml`
- `scripts/ci/comment_evaluation_report_pr.js`

## Changes

- `scripts/ci/check_workflow_publish_helper_adoption.py`
  - now emits richer summary counts:
    - `raw_publish_violation_count`
    - `missing_comment_helper_import_count`
    - `missing_issue_helper_import_count`
  - now supports `--output-md`
- `stress-tests.yml`
  - generates `workflow_publish_helper_adoption.json`
  - generates `workflow_publish_helper_adoption.md`
  - uploads artifact `workflow-publish-helper-adoption-${{ github.run_number }}`
  - appends markdown to `GITHUB_STEP_SUMMARY`
- `evaluation-report.yml`
  - builds optional `workflow_publish_helper_for_comment.json`
  - builds optional `workflow_publish_helper_for_comment.md`
  - appends markdown to `GITHUB_STEP_SUMMARY`
  - passes `WORKFLOW_PUBLISH_HELPER_SUMMARY_JSON_FOR_COMMENT` into PR comment script
- `comment_evaluation_report_pr.js`
  - adds `summarizeWorkflowPublishHelper(...)`
  - surfaces `Workflow Publish Helper Adoption` in Additional Analysis
  - surfaces `Workflow Publish Helper` in Signal Lights

## Validation

Commands:

```bash
pytest -q \
  tests/unit/test_check_workflow_publish_helper_adoption.py \
  tests/unit/test_stress_workflow_workflow_file_health.py \
  tests/unit/test_comment_evaluation_report_pr_js.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/unit/test_workflow_file_health_make_target.py

make validate-workflow-publish-helper-adoption-tests
make validate-workflow-file-health-tests
TMPDIR=$PWD/.tmp_pytest PYTEST_ADDOPTS="--basetemp=$PWD/.tmp_pytest/basetemp" make validate-eval-with-history-ci-workflows
TMPDIR=$PWD/.tmp_pytest PYTEST_ADDOPTS="--basetemp=$PWD/.tmp_pytest/basetemp" make validate-ci-watchers
```

Expected results:

- publish-helper guard emits JSON and markdown
- `stress-tests` wiring checks pass
- `evaluation-report` wiring checks pass
- PR comment runtime tests include publish-helper summary and parse-error cases
- total watcher gate remains green
