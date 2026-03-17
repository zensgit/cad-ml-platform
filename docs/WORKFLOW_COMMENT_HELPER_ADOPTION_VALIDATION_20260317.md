# Workflow Comment Helper Adoption Validation

Date: 2026-03-17
Branch: `feat/hybrid-blind-drift-autotune-e2e`

## Scope

Adopt the shared JS PR comment upsert helper in workflow-inline GitHub Script steps:

- `.github/workflows/release-risk-check.yml`
- `.github/workflows/pr-auto-label-comment.yml`
- `.github/workflows/sbom.yml`

Helper reused:

- `scripts/ci/comment_pr_utils.js`

## Changes

- `release-risk-check.yml`
  - replaced inline `listComments/updateComment/createComment` logic
  - now requires `./scripts/ci/comment_pr_utils.js`
  - now calls `upsertBotIssueComment(...)` with marker `Release Risk Assessment`
- `sbom.yml`
  - replaced inline `listComments/updateComment/createComment` logic
  - now requires `./scripts/ci/comment_pr_utils.js`
  - now calls `upsertBotIssueComment(...)` with marker `SBOM Dependency Changes`
- `pr-auto-label-comment.yml`
  - replaced inline `listComments/updateComment/createComment` logic
  - now requires `./scripts/ci/comment_pr_utils.js`
  - now calls `upsertBotIssueComment(...)` with its existing HTML marker
- Added workflow regression tests:
  - `tests/unit/test_release_risk_comment_workflow.py`
  - `tests/unit/test_pr_auto_label_comment_workflow.py`
  - `tests/unit/test_sbom_comment_workflow.py`
- Added Make target:
  - `validate-workflow-comment-helper-tests`
- Wired the new target into:
  - `validate-ci-watchers`

## Validation

Commands:

```bash
pytest -q \
  tests/unit/test_release_risk_comment_workflow.py \
  tests/unit/test_sbom_comment_workflow.py \
  tests/unit/test_workflow_file_health_make_target.py

make validate-workflow-comment-helper-tests
make validate-ci-watchers
```

Expected results:

- both workflow YAML regression tests pass
- new Make target resolves the expected test set
- `validate-ci-watchers` now includes workflow comment helper adoption checks

## Notes

- This change only touches workflow-inline comment publication logic.
- Comment body contents and label-management logic remain unchanged.
- No business/model code paths were modified.
