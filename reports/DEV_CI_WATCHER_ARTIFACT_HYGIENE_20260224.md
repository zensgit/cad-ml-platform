# CI Watcher Artifact Hygiene - Dev & Validation (2026-02-24)

## 1. Goal
- Prevent `reports/ci` runtime summaries from polluting git working tree.
- Keep CI watcher usage unchanged while improving day-to-day repo hygiene.

## 2. Implemented Changes

### 2.1 Ignore Runtime Summaries
Files:
- `.gitignore`
- `reports/ci/.gitkeep`

Changes:
- Ignore `reports/ci/*.json`.
- Keep directory trackable via `!reports/ci/.gitkeep`.

### 2.2 Add Cleanup Target
File:
- `Makefile`

Changes:
- Added `CI_WATCH_SUMMARY_DIR ?= reports/ci`.
- Added target:
  - `clean-ci-watch-summaries`
  - Removes `watch_commit_*_summary.json` under summary directory.

### 2.3 Docs Update
File:
- `README.md`

Changes:
- Clarified that watcher summary JSON under `reports/ci/` is runtime artifact and ignored by default.
- Added cleanup command example:
  - `make clean-ci-watch-summaries`

### 2.4 Test Update
File:
- `tests/unit/test_watch_commit_workflows_make_target.py`

Changes:
- Added make dry-run assertion for `clean-ci-watch-summaries` command pattern.

## 3. Validation

### 3.1 Unit Tests
```bash
pytest -q tests/unit/test_watch_commit_workflows.py tests/unit/test_watch_commit_workflows_make_target.py
```
Result:
- `20 passed, 1 warning`

### 3.2 Make Target Validation
```bash
make validate-watch-commit-workflows
```
Result:
- `20 passed`

### 3.3 Aggregated CI Watcher Validation
```bash
make validate-ci-watchers
```
Result:
- commit watcher suite: `20 passed`
- archive dispatcher suite: `24 passed`

### 3.4 Cleanup Target Smoke
```bash
make clean-ci-watch-summaries
```
Result:
- `reports/ci` only keeps `.gitkeep`.
