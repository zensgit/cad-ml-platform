# Evaluation Report Workflow Parse-Fix + PR Comment Module Validation (2026-03-15)

## Background
- `evaluation-report.yml` repeatedly failed with `workflow file issue` on push.
- Direct dispatch reproduced the parser error:
  - `failed to parse workflow: (Line: 2040, Col: 17): Exceeded max expression length 21000`

## Root Cause
- The `Comment PR with results` step embedded a very large inline `actions/github-script` script body.
- GitHub Actions parser hit expression-length limits on the workflow file.

## Changes Implemented

### 1) Extracted PR-comment logic into module
- Added:
  - `scripts/ci/comment_evaluation_report_pr.js`
- This module now owns:
  - score/status rendering
  - Graph2D/Hybrid signal generation
  - CI watcher summary parsing
  - PR comment upsert logic

### 2) Slimmed workflow inline script
- File: `.github/workflows/evaluation-report.yml`
- `Comment PR with results` now:
  - passes step outputs via `env`
  - calls module with a short script:
    - `require('./scripts/ci/comment_evaluation_report_pr.js')`
- Result: removed oversized inline script payload from workflow YAML.

### 3) Added workflow-file-health summary into PR comment
- Added global env:
  - `WORKFLOW_FILE_HEALTH_SUMMARY_JSON_FOR_COMMENT`
- Comment module now parses this JSON (if configured) and appends:
  - `Workflow File Health` in Additional Analysis
  - `Workflow Health` in Signal Lights

### 4) Updated regression tests
- File: `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
- Adjustments:
  - assert new global env key exists
  - assert comment step uses module entrypoint
  - assert comment step env wiring includes CI watcher + workflow health summary paths
  - assert semantic content checks against module source

## Local Validation

### Targeted workflow regression
```bash
pytest -q \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/unit/test_evaluation_report_workflow_hybrid_superpass_step.py \
  tests/unit/test_hybrid_superpass_workflow_integration.py
```
- Result: passed

### CI workflow regression suite
```bash
make validate-eval-with-history-ci-workflows
```
- Result: passed

### Full watcher stack regression
```bash
make validate-ci-watchers
```
- Result: passed

### JS syntax check
```bash
node --check scripts/ci/comment_evaluation_report_pr.js
```
- Result: passed

## Expected Remote Verification
- After push, run `gh workflow run ... evaluation-report.yml` should no longer return the parse-length 422 error.
- Existing non-blocking `.github/workflows/evaluation-report.yml` `workflow file issue` noise should clear if parser accepts new revision.
