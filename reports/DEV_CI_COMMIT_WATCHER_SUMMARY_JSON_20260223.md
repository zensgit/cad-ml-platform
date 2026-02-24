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
- Clarified that `CI_WATCH_SHA` supports `HEAD`/short SHA/full SHA and is normalized via `git rev-parse`.

### 2.5 SHA Resolution Fix
File: `scripts/ci/watch_commit_workflows.py`

- Fixed `resolve_head_sha` behavior for non-`HEAD` input:
  - previous behavior: short SHA was used as-is and failed exact match against `gh run list` full `headSha`.
  - current behavior: always resolve input through `git rev-parse <value>` and match with full SHA.
- Added unit coverage for:
  - short SHA expansion
  - empty input fallback to `HEAD`

## 3. Validation

### 3.1 Unit Tests
```bash
pytest -q tests/unit/test_watch_commit_workflows.py tests/unit/test_watch_commit_workflows_make_target.py
```
Result:
- `19 passed, 1 warning`

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
- `19 passed`

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
- commit watcher suite: `19 passed`
- archive dispatcher suite: `24 passed`

### 3.6 Runtime Smoke Check (short SHA)
```bash
make watch-commit-workflows \
  CI_WATCH_SHA=530897b \
  CI_WATCH_EVENTS=push \
  CI_WATCH_SUMMARY_JSON=reports/ci/watch_commit_530897b_shortsha_summary.json
```
Result:
- short SHA path successfully resolves and matches all workflows.
- run completed with `observed=11 completed=11 failed=0`.

## 4. Rollback
- Revert these files to rollback this feature:
  - `scripts/ci/watch_commit_workflows.py`
  - `Makefile`
  - `tests/unit/test_watch_commit_workflows.py`
  - `tests/unit/test_watch_commit_workflows_make_target.py`
  - `README.md`
