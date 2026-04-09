# Workflow Issue Helper Adoption Validation

Date: 2026-03-17
Branch: `feat/hybrid-blind-drift-autotune-e2e`

## Scope

Adopt a shared JS GitHub issue upsert helper in workflow-inline GitHub Script steps:

- `.github/workflows/badge-review.yml`
- `.github/workflows/adaptive-rate-limit-monitor.yml`

Helper:

- `scripts/ci/issue_upsert_utils.js`

## Changes

- Added `findOpenIssueByMarker(...)` and `upsertOpenIssue(...)`.
- `badge-review.yml`
  - monthly review issue now uses marker `<!-- ci:badge-review:${year}-${month} -->`
  - open issue is updated instead of creating duplicates for the same review period
- `adaptive-rate-limit-monitor.yml`
  - alert issue now uses marker `<!-- ci:adaptive-rate-limit-alert -->`
  - open issue is updated instead of creating repeated alert issues
- Added runtime tests:
  - `tests/unit/test_issue_upsert_utils_js.py`
- Added workflow regression tests:
  - `tests/unit/test_workflow_issue_helper_adoption.py`
- Added Make target:
  - `validate-workflow-issue-helper-tests`
- Wired the target into:
  - `validate-ci-watchers`

## Validation

Commands:

```bash
node --check scripts/ci/issue_upsert_utils.js
pytest -q \
  tests/unit/test_issue_upsert_utils_js.py \
  tests/unit/test_workflow_issue_helper_adoption.py \
  tests/unit/test_workflow_file_health_make_target.py

make validate-workflow-issue-helper-tests
make validate-ci-watchers
```

Expected results:

- shared helper syntax passes
- issue helper runtime tests pass
- workflow YAML adoption tests pass
- new Make target is included in `validate-ci-watchers`

## Notes

- This change only affects workflow-side GitHub issue publication behavior.
- It reduces duplicate open issues while preserving titles, labels, and body content.
