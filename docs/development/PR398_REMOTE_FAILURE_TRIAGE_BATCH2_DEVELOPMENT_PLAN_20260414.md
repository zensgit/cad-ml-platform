# PR398 Remote Failure Triage Batch 2 Development Plan

Date: 2026-04-14
PR: `#398`
Head: `submit/local-main-20260414`

## Scope

This batch addresses remote failures discovered after local unit baseline recovery:

1. `CI Tiered Tests` / `unit-tier`
2. `Security Audit`
3. `Adaptive Rate Limit Monitor`

## Findings

### 1. CI Tiered Tests

Remote run `24399473960` failed on:

- `tests/unit/test_comment_evaluation_report_pr_js.py::test_comment_evaluation_report_pr_js_runtime_body_matches_builder_output`

The failure only reproduced in GitHub-hosted execution, not in a clean local single-test run, which pointed to environment leakage into Node subprocesses.

### 2. Security Audit

Parallel read-only triage identified a `bandit` high-severity delta caused by:

- `src/ml/low_conf_queue.py`

The queue helper used `hashlib.md5(...)`, which tripped `B324`.

### 3. Adaptive Rate Limit Monitor

Parallel read-only triage identified a workflow structure bug:

- `.github/workflows/adaptive-rate-limit-monitor.yml`

The `post-pr-comment` job required `./scripts/ci/comment_pr_utils.js` but had no checkout step, so the GitHub-hosted runner could not resolve the module.

## Planned Changes

### A. Stabilize comment-evaluation-report Node tests

File:

- `tests/unit/test_comment_evaluation_report_pr_js.py`

Change:

- run Node subprocesses with a minimal clean environment instead of inheriting the full parent process environment

### B. Remove MD5 from low-confidence queue hashing

File:

- `src/ml/low_conf_queue.py`

Change:

- switch queue hash helper from MD5 to SHA-256 truncation
- remove the last `md5_hex` example identifier in the module docstring

### C. Fix adaptive-rate-limit PR comment job

Files:

- `.github/workflows/adaptive-rate-limit-monitor.yml`
- `tests/unit/test_additional_workflow_comment_helper_adoption.py`

Change:

- add `actions/checkout` to `post-pr-comment`
- add a regression assertion so this checkout requirement cannot silently disappear

## Out of Scope

These remain separate tracks:

- `Action Pin Guard` repo-wide SHA pin debt
- `PR Auto Label and Comment`
- broader security baseline debt outside `low_conf_queue.py`
