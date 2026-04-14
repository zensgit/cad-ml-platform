# PR398 Remote Failure Triage Batch 3 Development Plan

Date: 2026-04-14
Branch: `submit/local-main-20260414`
Scope: Isolated remote workflow failures on PR `#398`

## Background

After Batch 2, the main CI surfaces were green, but two isolated workflow failures remained:

1. `PR Auto Label and Comment`
   - Job: `label-and-comment`
   - Failed step: `Upsert PR context comment`
   - Symptom: `MODULE_NOT_FOUND` for `./scripts/ci/comment_pr_utils.js`
2. `Stress and Observability Checks`
   - Job: `workflow-file-health`
   - Failed step: `Generate workflow inventory audit report`
   - Symptom: step exited with code `2`

`Action Pin Guard` remained a separate repo-wide pin debt and is explicitly out of scope for this batch.

## Root Cause Summary

### 1. PR Auto Label and Comment

The workflow uses `actions/github-script` to `require('./scripts/ci/comment_pr_utils.js')`, but the job did not checkout the repository first. On GitHub-hosted runners, that makes local helper resolution fail.

Target file:
- `.github/workflows/pr-auto-label-comment.yml`

### 2. Stress and Observability Checks

The `workflow-file-health` job runs `scripts/ci/generate_workflow_inventory_report.py`, which requires `PyYAML`. The job only set up Python and did not install `pyyaml`, so the script returned `2` by design when `yaml` import was unavailable.

Target files:
- `.github/workflows/stress-tests.yml`
- `scripts/ci/generate_workflow_inventory_report.py`

## Planned Changes

1. Add pinned `actions/checkout` to `pr-auto-label-comment.yml` before the `github-script` steps.
2. Add an explicit `Install PyYAML` step to the `workflow-file-health` job in `stress-tests.yml`.
3. Extend workflow regression tests so both fixes are guarded locally:
   - `tests/unit/test_pr_auto_label_comment_workflow.py`
   - `tests/unit/test_stress_workflow_workflow_file_health.py`

## Acceptance Criteria

1. Local workflow regression tests pass.
2. Local workflow inventory generation still passes.
3. Sidecar review finds no issues in the narrow diff.
4. After push, the following remote workflows turn green on the new head:
   - `PR Auto Label and Comment`
   - `Stress and Observability Checks`

