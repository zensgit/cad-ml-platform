# Workflow Publish Helper Adoption Guard Validation

Date: 2026-03-17
Branch: `feat/hybrid-blind-drift-autotune-e2e`

## Scope

Add a static guard that blocks raw GitHub comment/issue publish calls from drifting back into workflow files.

Script:

- `scripts/ci/check_workflow_publish_helper_adoption.py`

## What It Checks

- forbids raw workflow-side GitHub publish calls:
  - `github.rest.issues.createComment(...)`
  - `github.rest.issues.updateComment(...)`
  - `github.rest.issues.listComments(...)`
  - `github.rest.issues.create(...)`
  - `github.rest.issues.update(...)`
  - `github.rest.issues.listForRepo(...)`
- requires comment helper import for designated workflows
  - `scripts/ci/comment_pr_utils.js`
- requires issue helper import for designated workflows
  - `scripts/ci/issue_upsert_utils.js`

## Changes

- Added guard script:
  - `scripts/ci/check_workflow_publish_helper_adoption.py`
- Added unit tests:
  - `tests/unit/test_check_workflow_publish_helper_adoption.py`
- Added Make targets:
  - `validate-workflow-publish-helper-adoption`
  - `validate-workflow-publish-helper-adoption-tests`
- Wired both targets into:
  - `validate-ci-watchers`

## Validation

Commands:

```bash
python3 scripts/ci/check_workflow_publish_helper_adoption.py \
  --workflow-root .github/workflows \
  --summary-json-out reports/ci/workflow_publish_helper_adoption.json

pytest -q \
  tests/unit/test_check_workflow_publish_helper_adoption.py \
  tests/unit/test_workflow_file_health_make_target.py

make validate-workflow-publish-helper-adoption
make validate-workflow-publish-helper-adoption-tests
make validate-ci-watchers
```

Expected results:

- guard passes on current repository state
- unit tests cover success and failure modes
- Make target assertions include the new guard
- `validate-ci-watchers` runs the guard and its tests
