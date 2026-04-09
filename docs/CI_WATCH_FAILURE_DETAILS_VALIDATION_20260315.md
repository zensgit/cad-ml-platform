# CI Watch Failure Details Validation (2026-03-15)

## Scope

Enhance commit workflow watcher to provide optional failure diagnostics, while keeping default behavior backward-compatible.

## Changes

1. `scripts/ci/watch_commit_workflows.py`
- Added optional flags:
  - `--print-failure-details`
  - `--failure-details-max-runs`
- Added failure detail retrieval via `gh run view --json url,workflowName,jobs`.
- Added extraction and output of failed jobs and first failed step per failed job.
- Kept default behavior unchanged when `--print-failure-details` is not enabled.

2. `Makefile`
- Added watcher vars:
  - `CI_WATCH_PRINT_FAILURE_DETAILS ?= 0`
  - `CI_WATCH_FAILURE_DETAILS_MAX_RUNS ?= 3`
- Added passthrough for:
  - `--failure-details-max-runs "$(CI_WATCH_FAILURE_DETAILS_MAX_RUNS)"`
  - optional `--print-failure-details` flag when enabled.

3. Tests
- Added `tests/unit/test_analyze_graph2d_gate_helpers.py`:
  - coverage for Graph2D enrich/filter and soft-override suggestion helper logic in `src/api/v1/analyze.py`.
- Updated `tests/unit/test_watch_commit_workflows.py`:
  - `get_run_failure_detail` parsing validation.
  - main flow validation for `--print-failure-details`.
- Updated `tests/unit/test_watch_commit_workflows_make_target.py`:
  - Make passthrough and flag rendering validation.

## Validation

Executed locally:

```bash
pytest -q tests/unit/test_analyze_graph2d_gate_helpers.py
```

Result:
- `5 passed`

```bash
pytest -q tests/unit/test_watch_commit_workflows.py tests/unit/test_watch_commit_workflows_make_target.py tests/unit/test_analyze_graph2d_gate_helpers.py
```

Result:
- `44 passed`

```bash
make validate-watch-commit-workflows
```

Result:
- `39 passed`

## Notes

- This enhancement improves failure triage speed after CI red runs.
- No breaking changes in existing watcher defaults.
