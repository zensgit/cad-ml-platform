# CI Watcher Safe Precheck Modes - Dev & Validation (2026-02-24)

## 1. Goal
- Improve `watch-commit-workflows-safe` usability under unstable network/auth conditions.
- Preserve strict default while enabling controlled non-strict troubleshooting mode.

## 2. Changes

### 2.1 Make Variables
File: `Makefile`

- Added:
  - `CI_WATCH_PRECHECK_STRICT ?= 1`
  - `GH_READY_JSON ?= reports/ci/gh_readiness_latest.json`
  - `GH_READY_SKIP_ACTIONS_API ?= 0`

### 2.2 Readiness Target
File: `Makefile`

- `check-gh-actions-ready` now supports:
  - writing JSON output via `GH_READY_JSON`
  - optional `--skip-actions-api` via `GH_READY_SKIP_ACTIONS_API=1`

### 2.3 Safe Watcher Target
File: `Makefile`

- `watch-commit-workflows-safe` behavior:
  - strict mode (`CI_WATCH_PRECHECK_STRICT=1`, default): precheck must pass
  - non-strict mode (`CI_WATCH_PRECHECK_STRICT=0`): precheck failure prints warning and continues

### 2.4 Script Update
File: `scripts/ci/check_gh_actions_ready.py`

- Added `--skip-actions-api`.
- JSON payload now includes `skip_actions_api` flag.

### 2.5 Tests
Files:
- `tests/unit/test_check_gh_actions_ready.py`
- `tests/unit/test_watch_commit_workflows_make_target.py`

Added/updated coverage:
- skip-actions-api path does not invoke `gh run list`
- payload includes `skip_actions_api=true`
- make dry-run includes new readiness JSON and skip flag logic
- make dry-run safe target includes strict-mode branch logic

## 3. Validation

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q \
  tests/unit/test_check_gh_actions_ready.py \
  tests/unit/test_watch_commit_workflows.py \
  tests/unit/test_watch_commit_workflows_make_target.py
```

```bash
make validate-ci-watchers
```

```bash
make -n watch-commit-workflows-safe
make -n check-gh-actions-ready
```
