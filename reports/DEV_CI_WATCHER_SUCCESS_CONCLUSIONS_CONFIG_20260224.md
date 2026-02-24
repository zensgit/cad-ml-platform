# CI Watcher Success Conclusions Config - Dev & Validation (2026-02-24)

## 1. Goal
- Make commit workflow watcher adaptable to different GitHub workflow conclusion policies.
- Avoid false negatives when a workflow returns `neutral` but should be accepted.

## 2. Changes

### 2.1 Script
File: `scripts/ci/watch_commit_workflows.py`

- Added CLI argument:
  - `--success-conclusions-csv` (default: `success,skipped`)
- Replaced hardcoded success conclusion set with configurable values.
- Added validation: empty configured list returns argument error (`exit 2`).
- Included `success_conclusions` in summary JSON payload.
- Included `success_conclusions` in `--print-only` preview output.

### 2.2 Makefile
File: `Makefile`

- Added variable:
  - `CI_WATCH_SUCCESS_CONCLUSIONS ?= success,skipped`
- Forwarded to watcher command:
  - `--success-conclusions-csv "$(CI_WATCH_SUCCESS_CONCLUSIONS)"`

### 2.3 Tests
Files:
- `tests/unit/test_watch_commit_workflows.py`
- `tests/unit/test_watch_commit_workflows_make_target.py`

Added/updated coverage:
- make target includes `--success-conclusions-csv`.
- print-only output includes normalized configured conclusions.
- empty `success_conclusions_csv` validation path (`exit 2`).
- configured `neutral` conclusion treated as success.
- summary JSON includes `success_conclusions`.

### 2.4 Docs
File: `README.md`

- Added usage of `CI_WATCH_SUCCESS_CONCLUSIONS`.
- Added guidance for workflows that return `neutral`.
- Added auth troubleshooting note (`gh auth login -h github.com`).

### 2.5 Auth Error Message Hardening
File: `scripts/ci/watch_commit_workflows.py`

- Improved `gh auth status` failure parsing so invalid-token scenarios return actionable details.
- Avoids generic `github.com` error text and includes re-authentication hints directly in watcher output.

## 3. Validation

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q \
  tests/unit/test_watch_commit_workflows.py \
  tests/unit/test_watch_commit_workflows_make_target.py
```

Expected:
- watcher suite passes with new config behavior and regression coverage.

Observed:
- `24 passed, 1 warning`
