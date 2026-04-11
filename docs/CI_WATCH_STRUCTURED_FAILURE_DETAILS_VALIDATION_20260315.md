# CI Watch Structured Failure Details Validation (2026-03-15)

## Scope

Further CI watcher hardening for faster failure triage and machine-consumable artifacts.

## Implemented

1. `scripts/ci/watch_commit_workflows.py`
- `summary.json` now always includes `failure_details` (list, default `[]`).
- On failed workflows:
  - base details (`run_id`, `workflow_name`, `conclusion`, `url`) are captured.
  - when `--print-failure-details` is enabled, enriched details are captured:
    - `failed_jobs`
    - `failed_steps`
    - optional `detail_unavailable` on gh query errors.
- `_log_failure_details_for_runs(...)` now returns structured detail rows while preserving console output.

2. `scripts/ci/generate_ci_watcher_validation_report.py`
- Added `repo` rendering in "Watch Summary Artifact".
- Added new "Failure Details" section:
  - renders structured `failure_details` rows when present
  - otherwise prints fallback line.

3. Tests
- `tests/unit/test_watch_commit_workflows.py`
  - added structured return assertion for failure detail collector
  - added summary assertions for `failure_details` in print-only/error paths
  - added failure-path summary test with structured detail payload
- `tests/unit/test_generate_ci_watcher_validation_report.py`
  - verifies `repo` rendering
  - verifies failure detail section rendering

## Validation

```bash
pytest -q tests/unit/test_watch_commit_workflows.py tests/unit/test_watch_commit_workflows_make_target.py
```
- Result: `44 passed`

```bash
pytest -q tests/unit/test_generate_ci_watcher_validation_report.py
```
- Result: `3 passed`

```bash
make validate-ci-watchers
```
- Result: all watcher-stack validations passed.

## Outcome

- CI failure diagnostics are now both human-readable and machine-serializable.
- Downstream tooling can parse `failure_details` directly from watcher summaries without additional `gh run view` calls.
