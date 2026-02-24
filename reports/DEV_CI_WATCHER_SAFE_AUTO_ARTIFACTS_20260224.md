# CI Watcher Safe Auto Artifacts - Dev & Validation (2026-02-24)

## 1. Goal
- Reduce manual parameter overhead for `watch-commit-workflows-safe`.
- Ensure readiness/watch artifacts are always generated with commit-specific names.

## 2. Changes

### 2.1 New Make Target
File: `Makefile`

- Added target:
  - `watch-commit-workflows-safe-auto`
- Behavior:
  - resolves target commit with `git rev-parse "$(CI_WATCH_SHA)"`
  - auto-derives artifact paths:
    - `$(CI_WATCH_SUMMARY_DIR)/gh_readiness_watch_<sha>.json`
    - `$(CI_WATCH_SUMMARY_DIR)/watch_commit_<sha>_summary.json`
  - invokes `watch-commit-workflows-safe` with computed paths

### 2.2 Tests
File: `tests/unit/test_watch_commit_workflows_make_target.py`

- Added dry-run test:
  - validates `git rev-parse` call
  - validates auto artifact filename patterns
  - validates chained call to `watch-commit-workflows-safe`

### 2.3 Docs
File: `README.md`

- Added usage example for:
  - `make watch-commit-workflows-safe-auto`

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

```bash
make -n watch-commit-workflows-safe-auto
```
