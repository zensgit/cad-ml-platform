# Commit Workflow Watcher Summary JSON - Dev & Validation (2026-02-23)

## 1. Scope
- Enhance `scripts/ci/watch_commit_workflows.py` with machine-readable final output for automation/report pipelines.
- Keep existing watcher behavior unchanged for normal console usage.

## 2. Changes Implemented

### 2.1 Script Enhancement
File: `scripts/ci/watch_commit_workflows.py`

- Added CLI arg:
  - `--summary-json-out` (optional): writes final execution summary as JSON.
- Added structured summary payload fields:
  - `requested_sha`, `resolved_sha`, `events`, `required_workflows`
  - `missing_required_mode`, `failure_mode`
  - `wait_timeout_seconds`, `poll_interval_seconds`, `heartbeat_interval_seconds`, `list_limit`
  - `exit_code`, `reason`, `started_at_unix`, `finished_at_unix`, `duration_seconds`
  - `counts` (`observed/completed/failed/missing_required`)
  - `missing_required`, `failed_workflows`, `runs` (workflow snapshots)
- Added unified return path to ensure JSON summary is written consistently on:
  - success
  - fail-fast failures
  - timeout failures
  - runtime failures (`gh`/SHA resolve/list errors)
  - `--print-only` mode

### 2.2 Makefile Integration
File: `Makefile`

- Added variable:
  - `CI_WATCH_SUMMARY_JSON ?=`
- Updated target:
  - `watch-commit-workflows` now always forwards:
    - `--summary-json-out "$(CI_WATCH_SUMMARY_JSON)"`

### 2.3 Tests Updated
Files:
- `tests/unit/test_watch_commit_workflows.py`
- `tests/unit/test_watch_commit_workflows_make_target.py`

Added coverage for:
- summary JSON written in print-only path
- write failure handling returns non-zero
- Make target includes new `--summary-json-out` flag

### 2.4 Docs Updated
File: `README.md`

- Added `CI_WATCH_SUMMARY_JSON` usage example and behavior notes in watcher section.

## 3. Validation

### 3.1 Unit Tests
```bash
pytest -q tests/unit/test_watch_commit_workflows.py tests/unit/test_watch_commit_workflows_make_target.py
```
Result:
- `17 passed, 1 warning`

### 3.2 Lint (changed files)
```bash
python3 -m flake8 scripts/ci/watch_commit_workflows.py tests/unit/test_watch_commit_workflows.py tests/unit/test_watch_commit_workflows_make_target.py --max-line-length=100
```
Result:
- pass

### 3.3 Make Target Validation
```bash
make validate-watch-commit-workflows
```
Result:
- `17 passed`

### 3.4 Runtime Smoke Check (print-only + JSON)
```bash
python3 scripts/ci/watch_commit_workflows.py --print-only --summary-json-out /tmp/ci-watch-summary-test.json
make watch-commit-workflows CI_WATCH_PRINT_ONLY=1 CI_WATCH_SUMMARY_JSON=/tmp/ci-watch-summary-from-make.json
```
Result:
- command preview正常输出
- summary JSON文件成功落盘

### 3.5 Aggregated Watcher Validation
```bash
make validate-ci-watchers
```
Result:
- commit watcher suite: `17 passed`
- archive dispatcher suite: `24 passed`

## 4. Rollback
- Revert these files to rollback this feature:
  - `scripts/ci/watch_commit_workflows.py`
  - `Makefile`
  - `tests/unit/test_watch_commit_workflows.py`
  - `tests/unit/test_watch_commit_workflows_make_target.py`
  - `README.md`
