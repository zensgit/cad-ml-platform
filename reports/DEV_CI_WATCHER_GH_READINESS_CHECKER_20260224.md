# CI Watcher GH Readiness Checker - Dev & Validation (2026-02-24)

## 1. Goal
- Add a deterministic preflight check for `gh` availability, auth validity, and Actions API reachability.
- Reduce iteration cost when watcher fails due to environment issues.

## 2. Changes

### 2.1 New Readiness Checker Script
File: `scripts/ci/check_gh_actions_ready.py`

- Checks:
  - `gh --version`
  - `gh auth status`
  - `gh run list --limit 1`
- Output:
  - human-readable per-check status lines
  - optional JSON report via `--json-out`
- Exit code:
  - `0` when all checks pass
  - `1` when any check fails

### 2.2 Make Targets
File: `Makefile`

- Added:
  - `check-gh-actions-ready`
  - `validate-check-gh-actions-ready`
- Added convenience wrapper:
  - `watch-commit-workflows-safe` (runs readiness precheck before watcher)
- `validate-ci-watchers` now includes `validate-check-gh-actions-ready` first.

### 2.3 Test Coverage
File: `tests/unit/test_check_gh_actions_ready.py`

Covered scenarios:
- all checks pass + JSON output is written
- auth invalid failure includes actionable guidance
- API connectivity failure is surfaced clearly
- error extraction prefers actionable lines

### 2.4 Watcher Auth Diagnostics Hardening
File: `scripts/ci/watch_commit_workflows.py`

- Improved auth error parsing for `gh auth status` failures.
- Invalid token path now emits actionable message including `gh auth login -h github.com`.

### 2.5 Docs
File: `README.md`

- Added `make check-gh-actions-ready` command in watcher section.
- Updated `validate-ci-watchers` composition to include readiness checker validation.

## 3. Validation

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q \
  tests/unit/test_check_gh_actions_ready.py \
  tests/unit/test_watch_commit_workflows.py \
  tests/unit/test_watch_commit_workflows_make_target.py
```

Observed:
- all watcher/readiness tests pass

```bash
make validate-ci-watchers
```

Observed:
- readiness checker unit suite: pass
- commit watcher suite: pass
- archive dispatcher suite: pass
