# CI Watcher Artifact Cleanup Expansion - Dev & Validation (2026-02-24)

## 1. Goal
- Extend cleanup coverage from `watch_commit_*` summaries to GH readiness JSON artifacts.
- Provide one-shot cleanup entry for all CI watcher runtime JSON outputs.

## 2. Changes

### 2.1 Make Targets
File: `Makefile`

- Added:
  - `clean-gh-readiness-summaries`
    - removes `$(CI_WATCH_SUMMARY_DIR)/gh_readiness*.json`
  - `clean-ci-watch-artifacts`
    - calls both `clean-ci-watch-summaries` and `clean-gh-readiness-summaries`
- Updated:
  - `clean-ci-watch-summaries` now removes `watch_*_summary.json`
  - covers both commit watcher and safe watcher runtime summary files

### 2.2 Tests
File: `tests/unit/test_watch_commit_workflows_make_target.py`

Added coverage:
- dry-run includes `gh_readiness*.json` cleanup pattern
- aggregate cleanup target invokes both child targets

### 2.3 Docs
File: `README.md`

Cleanup section now includes:
- `make clean-ci-watch-summaries`
- `make clean-gh-readiness-summaries`
- `make clean-ci-watch-artifacts`

## 3. Validation

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q \
  tests/unit/test_watch_commit_workflows_make_target.py \
  tests/unit/test_watch_commit_workflows.py \
  tests/unit/test_check_gh_actions_ready.py
```

```bash
make validate-ci-watchers
```
