# Evaluation Comment Support Artifacts Validation

## Scope

Package the optional `evaluation-report` comment-support summaries into a single downloadable
artifact so reviewers can inspect the same JSON/Markdown inputs used by:

- PR comment summary rows
- PR comment signal lights
- appended `GITHUB_STEP_SUMMARY` sections

## Files

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Change

Added `Upload evaluation comment support artifacts` to `evaluation-report.yml`.

Uploaded bundle contents:

- `reports/ci/workflow_file_health_for_comment.json`
- `reports/ci/workflow_inventory_for_comment.json`
- `reports/ci/workflow_inventory_for_comment.md`
- `reports/ci/workflow_publish_helper_for_comment.json`
- `reports/ci/workflow_publish_helper_for_comment.md`
- `reports/ci/workflow_guardrail_for_comment.json`
- `reports/ci/workflow_guardrail_for_comment.md`
- `reports/ci/ci_workflow_guardrail_overview_for_comment.json`
- `reports/ci/ci_workflow_guardrail_overview_for_comment.md`
- `reports/ci/ci_watch_validation_for_comment.json`
- `reports/ci/ci_watch_validation_for_comment.md`

Artifact settings:

- name: `evaluation-comment-support-${{ github.run_number }}`
- `if-no-files-found: ignore`
- `retention-days: ${{ env.ARTIFACT_RETENTION_DAYS }}`

## Validation

Commands run:

```bash
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
TMPDIR=$PWD/.tmp_pytest PYTEST_ADDOPTS="--basetemp=$PWD/.tmp_pytest/basetemp" make validate-eval-with-history-ci-workflows
```

## Outcome

`evaluation-report` now exposes the full comment-supporting summary bundle as one artifact, so the
review surface and the underlying structured inputs stay aligned and downloadable.
