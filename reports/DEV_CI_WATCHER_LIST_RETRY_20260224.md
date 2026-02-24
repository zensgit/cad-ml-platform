# CI Watcher Run-List Retry - Dev & Validation (2026-02-24)

## 1. Goal
- Improve watcher resilience against transient `gh run list` failures (network/API jitter).
- Avoid immediate false failures for short-lived connectivity issues.

## 2. Changes

### 2.1 Watcher Retry Logic
File: `scripts/ci/watch_commit_workflows.py`

- Added CLI argument:
  - `--max-list-failures` (default: `3`)
- Behavior:
  - transient `gh run list` failures increment consecutive failure counter
  - watcher retries until consecutive failures exceed threshold
  - abort reason on exceed: `gh_run_list_failed`
  - timeout during retry loop: `timeout_list_runs`
- Summary JSON now includes:
  - `max_list_failures`
  - `consecutive_list_failures`

### 2.2 Makefile
File: `Makefile`

- Added variable:
  - `CI_WATCH_MAX_LIST_FAILURES ?= 3`
- Forwarded to watcher command:
  - `--max-list-failures "$(CI_WATCH_MAX_LIST_FAILURES)"`

### 2.3 Tests
Files:
- `tests/unit/test_watch_commit_workflows.py`
- `tests/unit/test_watch_commit_workflows_make_target.py`

Added/updated coverage:
- make target includes `--max-list-failures`
- print-only output includes configured max list failures
- argument validation for negative value
- retry then success path
- exceeding retry budget path
- summary JSON includes retry fields

### 2.4 Docs
File: `README.md`

- Added `CI_WATCH_MAX_LIST_FAILURES` usage and explanation.

## 3. Validation

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q \
  tests/unit/test_check_gh_actions_ready.py \
  tests/unit/test_watch_commit_workflows.py \
  tests/unit/test_watch_commit_workflows_make_target.py
```

Expected:
- all watcher/readiness tests pass.

```bash
make validate-ci-watchers
```

Expected:
- readiness checker tests pass
- commit watcher tests pass
- archive dispatcher tests pass
